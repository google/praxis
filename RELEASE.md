# Version: 1.3.1
## Major Features and Improvements
## Breaking changes
## Deprecations
## Note
*   Version: 1.3.1
*   Build Date: 20240220
*   Praxis commit: a935384da88ec739d48696c2d0a55657a183de00
# Version: 1.3.0
## Major Features and Improvements
## Breaking changes
## Deprecations
## Note
*   Version: 1.3.0
*   Build Date: 20240216
*   Praxis commit: ca8ad2ba8be5b77092a8e5f0a0774a39cc3ad766
# Version: 1.2.0
## Major Features and Improvements
## Breaking changes
## Deprecations
## Note
*   Version: 1.2.0
*   Build Date: 20231016
*   Praxis commit: 7bd63412bf86a68e09fcd9455f76a4909d19377e
# Version: 1.1.0
## Major Features and Improvements
* Move to python 3.10 as the minimal python requirement (previously on python 3.8).
* Add various quantization support.
* Add support for fine-grained weight sparsity in praxis layers.
* Add support for Direct Preference Optimization (DPO).
## Note
*   Version: 1.1.0
*   Build Date: 20230824
*   Praxis commit: 2a7d0407871502a1d79dcd0e01411e73f1d15d36
# Version: 1.0.0
## Major Features and Improvements
* **Fiddle** - Praxis layers and BaseParameterizable are now configured with [Fiddle](https://github.com/google/fiddle), a Python-first configuration library. Fiddle reduces boilerplate, and adds productivity features including history tracking, graphviz visualization, support for aliasing objects, and more.
* **CLI Experiment and Data Injectability** - Enable Pax users to select which experiments to run without the need to recompile for each experiment.  Using a CLI interface based on Fiddle, users can override subsets of the experiment’s canonical dataset.
* **CLU Metrics** - Praxis has adopted CLU metrics as its standard metric interface.  This allows other Jax/Flax codebases that have CLU metrics to use them in Praxis.
* **Flax Interoperability** - Praxis now supports shape inference, __call__ for forward propagation, and has adopted Linen’s AxisMetadata for its mid-level sharding APIs.  These changes improve interoperability with other Flax-based libraries such as T5X.
## Note
*   Version: 1.0.0
*   Build Date: 20230329
*   Praxis commit: 621c2ca7bfcd0e21ea118a3d8e40e29b48313c0c
# Version: 0.4.0
## Note
*   Version: 0.4.0
*   Build Date: 20230329
*   Praxis commit: 621c2ca7bfcd0e21ea118a3d8e40e29b48313c0c
# Version: 0.3.0
## Major Features and Improvements
* Fiddle migration
* Improve numerical stability when using bfloat16
* Improve and add new functionalities to decoding algorithms
* Improve quantization support and add quantization aware training
* Improve streaming support
* Move learners / sgf and train_states modules to paxml
* Misc renaming / API updates for consistency
## Note
*   Version: 0.3.0
*   Build Date: 20230201
*   Praxis commit: 9e1d13d888ac18a567e249ddb41e6b1bd1fe505a
# Version: 0.2.1
## Note
*   Version: 0.2.1
*   Build Date: 20221121
*   Praxis commit: f7e98026c1c5ecbc6e4aff175621d443fa37fcf2
# Version: 0.2.0
## Major Features and Improvements
*  Preparatory work for Fiddle integration
*  Support for Flax shape inference
*  Support for Jax Array
*  Optimizer additions and improvements:
   - HeroLion
   - ShardedAdagrad
   - ShardedStaticAccumulator optimizer wrapper to do a fixed number of gradient
     accumulations
   - Shampoo improvements
   - Fix for multi-optimizer following the introduction of optax.MaskedNode
   - Improve sanitization of NaNs/Infs gradients during training
* Decoding
   - Add support for ExtendNSteps
   - Add beam search support for sequence models
   - Set prefix_lengths by input_indicator for PrefixLM
   - Move decode post-processing tensors into host memory
* Summaries
   - Add support for verbosity level
   - Add more knobs to the learner to control summary generation
## Deprecations
*  Disallow hparams override in setup()
*  Hparams and layer names must now be distinct
## Note
*   Version: 0.2.0
*   Build Date: 20221114
*   Praxis commit: 413da1ad8148f27faebca119f8c5deedca66228b
# Version: 0.1.0
## Major Features and Improvements
## Breaking changes
## Deprecations
## Note
*   Version: 0.1.0
*   Build Date: 20220702
*   Commit:
