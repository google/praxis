# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Linear layers."""

from jax import numpy as jnp
from jax import vmap
from jax.ad_checkpoint import checkpoint_name
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import base_ops


NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
JTensor = pytypes.JTensor
SplitDimsMapping = pytypes.SplitDimsMapping


def project_last_dim(
    inputs: JTensor,
    weight: JTensor,
) -> JTensor:
  """Linear projection on the last dim of the input JTensor.

  This is a TPU efficient implementation to avoid reshaping inputs to Rank-2
  JTensor by using Einsum for the compute.

  Args:
    inputs: An input JTensor, the last dimension of which is input_dim.
    weight: A weight matrix with shape [input_dim, output_dim].

  Returns:
    An output JTensor of the same rank as inputs, the last dimension is
    output_dim.
  """
  input_shape = inputs.shape
  assert len(input_shape) >= 2
  weight_shape = weight.shape
  assert len(weight_shape) == 2
  assert input_shape[-1] == weight_shape[0], (
      f'input_shape[-1] = {input_shape[-1]}, '
      f'weight_shape[0] = {weight_shape[0]}')
  return jnp.einsum('...y,yz->...z', inputs, weight)


class Linear(base_layer.BaseLayer):
  """Linear layer without bias.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
  """
  input_dims: int = 0
  output_dims: int = 0
  weight_init: WeightInit | None = None
  einsum_tpl: LayerTpl = template_field(base_ops.EinsumOp)

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      extend_step_out: Sharding annotations for the primary extend step output.
    """

    extend_step_out: SplitDimsMapping = None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    self.create_variable(
        'w',
        WeightHParams(
            shape=[self.input_dims, self.output_dims],
            init=self.weight_init,
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
        ),
    )
    self.create_child('einsum', self.einsum_tpl.clone())

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    ap = self.activation_split_dims_mapping
    out = self.einsum('...y,yz->...z', inputs, self.theta.w)
    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = ap.out
    if out.ndim == 2:
      if (
          hasattr(ap, 'extend_step_out')
          and ap.extend_step_out is not None
          and len(ap.extend_step_out) == 2
      ):
        ap_out = ap.extend_step_out
      elif ap_out is not None and len(ap_out) == 3:
        ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)
    return out


class Bias(base_layer.BaseLayer):
  """Bias layer.

  Attributes:
    dims: Depth of the input.
    bias_init: Init scale (constant) of bias terms.
  """
  dims: int = 0
  bias_init: float | None = 0.0

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    self.create_variable(
        'b',
        WeightHParams(
            shape=[self.dims],
            init=WeightInit.Constant(self.bias_init),
            mesh_shape=self.mesh_shape,
            tensor_split_dims_mapping=wp.wt,
        ),
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Adds bias to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., dims].

    Returns:
      Inputs plus bias.
    """
    return inputs + self.theta.b


class FeedForward(base_layer.BaseLayer):
  """Feedforward layer with activation.

  Attributes:
    input_dims: Depth of the input.
    output_dims: Depth of the output.
    has_bias: Adds bias weights or not.
    linear_tpl: Linear layer params.
    activation_tpl: Activation layer params.
    bias_init: Init scale (constant) of bias terms.
    checkpoint_str: name to checkpoint the tensor output from nn.linear.
  """
  input_dims: int = 0
  output_dims: int = 0
  has_bias: bool = True
  linear_tpl: LayerTpl = template_field(Linear)
  bias_tpl: LayerTpl = template_field(Bias)
  activation_tpl: pax_fiddle.Config[
      activations.BaseActivation
  ] = template_field(activations.ReLU)
  weight_init: WeightInit | None = None
  bias_init: float | None = 0.0
  checkpoint_str: str | None = None

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      extend_step_out: Sharding annotations for the primary extend step output.
    """

    extend_step_out: SplitDimsMapping = None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping
    linear_layer_p = self.linear_tpl.clone()
    linear_layer_p.set(
        input_dims=self.input_dims,
        output_dims=self.output_dims,
        weight_init=self.weight_init,
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone(),
    )
    # Provide type hint.
    self.linear: Linear
    self.create_child('linear', linear_layer_p)
    if self.has_bias:
      bias_layer_p = self.bias_tpl.clone()
      bias_layer_p.set(dims=self.output_dims, bias_init=self.bias_init
                       )
      if self.mesh_shape is not None and ap.out is not None:
        wp_bias = [ap.out[-1]]
        bias_layer_p.weight_split_dims_mapping.wt = wp_bias
      # Provide type hint.
      self.bias: Bias
      self.create_child('bias', bias_layer_p)
    # Provide type hints
    self.activation: activations.BaseActivation
    self.create_child('activation', self.activation_tpl.clone())

  def __call__(self, inputs: JTensor) -> JTensor:
    projected_inputs = self.linear(inputs)
    if self.checkpoint_str is not None:
      projected_inputs = checkpoint_name(projected_inputs, self.checkpoint_str)
    if self.has_bias:
      projected_inputs = self.bias(projected_inputs)
    output = self.activation(projected_inputs)
    return output


class MLPBlock(base_layer.BaseLayer):
  """Multilayer perceptron block composed of multiple FeedForward layers.

  Attributes:
    num_layers: Number of FeedForward layers.
    hidden_dims: Dimension of hidden layers.
    activate_final: If False, the activation layer is not applied after the last
      layer.
    ff_tpl: Feedforward layer params.
  """
  num_layers: int = 3
  hidden_dims: int = 128
  activate_final: bool = True
  ff_tpl: LayerTpl = template_field(FeedForward)

  def setup(self) -> None:

    if self.num_layers < 1:
      raise ValueError('num_layers must be at least 1.')

    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping
    hidden_layer_p = self.ff_tpl.clone()
    hidden_layer_p.set(
        input_dims=self.hidden_dims,
        output_dims=self.hidden_dims,
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone(),
    )
    mlp_layers = [hidden_layer_p.clone() for _ in range(self.num_layers)]
    mlp_layers[0].set(input_dims=self.ff_tpl.input_dims)
    mlp_layers[-1].set(output_dims=self.ff_tpl.output_dims)
    if not self.activate_final:
      mlp_layers[-1].set(activation_tpl=pax_fiddle.Config(activations.Identity))
    self.create_children('mlp_layers', mlp_layers)

  def __call__(self, inputs: JTensor) -> JTensor:
    output = inputs
    for i in range(self.num_layers):
      output = self.mlp_layers[i](output)
    return output


class StackingOverTime(base_layer.BaseLayer):
  """Stacking applied along the time axis.

  At each time step of an input sequence, elements are stacked over the
  window of ('left_context' + 1 + 'right_context') steps around the current
  time step. Zeros will be padded to the left or right of the sequence for
  elements around the boundaries. Finally the stacked outputs are emitted
  once every 'stride' steps.

  E.g. if an input sequence is: [4], [1], [9], [3], [5], [2], [8]
  left_context = 1, right_context = 1, stride = 3,
  then the output sequence would be: [0, 4, 1], [9, 3, 5], [2, 8, 0]

  Note that this layer only performs tensor transformation, so there are no
  learnable parameters.

  Attributes:
    left_context: Number of time steps to stack on the left to the central
      step.
    right_context: Number of time steps to stack on the right to the central
      step.
    stride: The stride for emitting the stacked output.
    pad_with_left_frame: Whether to use the left frame for padding instead of
      0s.
    pad_with_right_frame: Whether to use the right frame for padding instead
      of 0s.
    padding_reduce_option: reduce_max or reduce_min. How to reduce stacked
      padding from [b, t / stride, stride] to [b, t / stride, 1].
  """
  left_context: int = 0
  right_context: int = 0
  stride: int = 0
  pad_with_left_frame: bool = False
  pad_with_right_frame: bool = False
  padding_reduce_option: str = 'reduce_min'

  def setup(self) -> None:
    assert self.name
    assert self.left_context >= 0, self.left_context
    assert self.right_context >= 0, self.right_context
    assert self.stride >= 1
    assert self.padding_reduce_option in ('reduce_min', 'reduce_max')

  @property
  def window_size(self):
    """Returns the stacking window size.

    The output dimension will be window_size * the input dimension.

    Returns:
      Window size.
    """
    return self.left_context + self.right_context + 1

  def _pad_ends(self, inputs, pad_value):
    """Applies left and right padding to inputs."""
    left_to_pad = self.left_context
    right_to_pad = self.right_context

    # optionally copy left frame N times
    if left_to_pad and self.pad_with_left_frame:
      left_pad = jnp.repeat(inputs[:, :1, :], repeats=left_to_pad, axis=1)
      inputs = jnp.concatenate([left_pad, inputs], axis=1)
      left_to_pad = 0

    # optionally copy right frame N times
    if right_to_pad and self.pad_with_right_frame:
      right_pad = jnp.repeat(inputs[:, -1:, :], repeats=right_to_pad, axis=1)
      inputs = jnp.concatenate([inputs, right_pad], axis=1)
      right_to_pad = 0

    # Add zero paddings to the left and right of the input sequence.
    if left_to_pad or right_to_pad:
      inputs = jnp.pad(
          inputs, [[0, 0], [left_to_pad, right_to_pad], [0, 0]],
          constant_values=pad_value)

    return inputs

  def _apply_stack(self, inputs):
    """The core function to apply the stacking to inputs.

    Args:
      inputs: [batch, timesteps, depth].

    Returns:
      [batch, out_timesteps, window_size * depth] tensor.
    """
    if self.left_context == 0 and self.right_context == 0:
      out = inputs
    else:
      # Slide a window of size stack_width to extract window_size() slices from
      # the inputs, each offset by a single timestep. In non-streaming mode
      # stack_width is equivalent to the length of the unpadded input.
      stack_width = inputs.shape[1] - self.window_size + 1
      pieces = []
      for i in range(self.window_size):
        pieces.append(inputs[:, i:i + stack_width])
      # Apply stacking.
      out = jnp.concatenate(pieces, 2)

    # Apply striding.
    if self.stride > 1:
      out = out[:, :: self.stride]
    return out

  def _stack_inputs(self, inputs, pad_ends: bool = True):
    # Checks the inputs have 3 dims.
    base_layer.assert_has_shape(inputs, [-1, -1, -1])
    if pad_ends:
      inputs = self._pad_ends(inputs, pad_value=0.0)
    return self._apply_stack(inputs)

  def _stack_paddings(self, outputs, paddings, pad_ends: bool = True):
    if paddings is None:
      out_paddings = jnp.zeros(outputs.shape[:-1] + (1,), dtype=outputs.dtype)
    else:
      # Check that the paddings have 3 dims and a singleton final dim.
      base_layer.assert_has_shape(paddings, [-1, -1, 1])
      if pad_ends:
        paddings = self._pad_ends(paddings, pad_value=1.0)
      out_paddings = self._apply_stack(paddings)

      # The default is to take the minimum padding values within each stacking
      # window, so that an output time step becomes a padded one only if all of
      # the underlying stacked steps are padded ones.
      if self.padding_reduce_option == 'reduce_min':
        out_paddings = jnp.amin(out_paddings, axis=2, keepdims=True)
      else:
        out_paddings = jnp.amax(out_paddings, axis=2, keepdims=True)
    return out_paddings

  def __call__(self, inputs, paddings=None):
    """Apply the stacking to inputs along the time axis.

    Args:
      inputs: The inputs tensor with shape [batch, time, depth].
      paddings: The paddings tensor with shape [batch, time, 1], where all but
        the last dimension match inputs. Each value is 0 or 1 indicating whether
        a time step of a sequence is padded in the inputs to reach the max
        length in the batch.

    Returns:
      (outputs, out_paddings) pair.
        outputs is of shape [batch, ceil(time / stride), window_size * depth].
        out_paddings is of shape [batch, ceil(time / stride), 1]. out_paddings
        will be 0 if any of the corresponding input padding is 0.
    """
    # Trivial case.
    if 0 == self.left_context == self.right_context and 1 == self.stride:
      return inputs, paddings
    outputs = self._stack_inputs(inputs, pad_ends=True)
    out_paddings = self._stack_paddings(outputs, paddings, pad_ends=True)
    return outputs, out_paddings

  def unstack(self, stacked):
    """Inverts stacking over time.

    Given 'stacked' outputs from this StackingOverTime layer,

      stacked, _ = this_layer.FProp(inputs),

    this method attempts to reconstruct the original 'inputs'.

    If stride > window_size, the original input cannot be recovered, and a
    ValueError is raised.

    Otherwise, if right_context + 1 >= stride, this method returns a Tensor that
    is identical to 'inputs' but potentially longer due to paddings.

    If right_context + 1 < stride, this method returns a Tensor that may be up
    to ```stride - right_context - 1``` frames shorter than the original input,
    but identical in the frames that are returned. e.g.::

      left_context = 2, right_context = 1, stride = 4
      input sequence:     1 2 3 4 5 6 7 8
      after padding:  0 0 1 2 3 4 5 6 7 8 0
      windows:
        [0 0 (1) 2] 3 4 5 6 7 8 0
         0 0 1 2 [3 4 (5) 6] 7 8 0
      stacked:
        [[0 0 1 2], [3 4 5 6]]
      unstacked:
        [1 2 3 4 5 6], which is 4 - 1 - 1 = 2 (stride - right_context - 1)
        frames shorter than the original input.

    `unstack()` can be used to project the outputs of downstream layers back to
    the shape of the original unstacked inputs. For example::

        inputs = ...  # [batch, length, input_dim]
        # [batch, ceil(length / stride), rnn_dim]
        rnn_out = rnn(stacking(inputs)[0])
        # [batch, length, rnn_dim]
        back_projected_rnn_out = py_utils.PadOrTrimTo(
            stacking.unstack(jnp.tile(rnn_out, [1, 1, stacking.window_size])),
            inputs.shape)

    Note this method does not take or return a separate padding JTensor. The
    caller is responsible for knowing which of outputs are padding (e.g. based
    on the padding of the original FProp inputs).

    Args:
      stacked: JTensor of shape [batch, time, window_size * feature_dim],
        assumed to be the output of `fprop`.

    Returns:
      The reconstructed input JTensor, with shape
      [batch, (frames - 1) * stride + right_context + 1, feature_dim].

    Raises:
      ValueError: if stride > window_size.
    """
    if 0 == self.left_context == self.right_context and 1 == self.stride:
      return stacked

    if self.stride > self.window_size:
      raise ValueError(
          "Can't invert StackingOverTime with stride (%d) > window_size (%d)"
          % (self.stride, self.window_size)
      )

    # Reshape to allow indexing individual frames within each stacked window.
    batch_size, stacked_length, _ = stacked.shape
    stacked = jnp.reshape(stacked,
                          [batch_size, stacked_length, self.window_size, -1])

    # Compute the index of the window and frame in 'stacked' where each frame of
    # the original input is located, and extract them with tf.gather_nd.
    # First compute for all except the last window, since these elements have
    # the potential of being looked up from the next window.
    input_indices = jnp.arange(0, (stacked_length - 1) * self.stride)
    mod = input_indices % self.stride
    in_next_window = jnp.greater(mod, self.right_context).astype(jnp.int32)
    window_index = input_indices // self.stride + in_next_window
    frame_index = self.left_context + mod - self.stride * in_next_window
    # Now handle the last window explicitly and concatenate onto the existing
    # window_index/frame_index tensors.
    last_window_length = self.right_context + 1
    window_index = jnp.concatenate([
        window_index,
        jnp.repeat(jnp.array([stacked_length - 1]), last_window_length)
    ],
                                   axis=0)
    frame_index = jnp.concatenate(
        [frame_index, self.left_context + jnp.arange(last_window_length)],
        axis=0,
    )
    # Stack the indices for gather_nd operation below
    window_and_frame_indices = jnp.stack([window_index, frame_index], axis=1)
    window_and_frame_indices = jnp.tile(
        jnp.expand_dims(window_and_frame_indices, 0), [batch_size, 1, 1])

    # jax equivalent of tf.gather_nd
    def gather_nd_unbatched(params, indices):
      return params[tuple(jnp.moveaxis(indices, -1, 0))]

    return vmap(gather_nd_unbatched, (0, 0), 0)(stacked,
                                                window_and_frame_indices)
