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
