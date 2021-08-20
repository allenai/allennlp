# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added a module `allennlp.nn.parallel` with a new base class, `DdpAccelerator`, which generalizes
  PyTorch's `DistributedDataParallel` wrapper to support other implementations. Two implementations of
  this class are provided. The default is `TorchDdpAccelerator` (registered at "torch"), which is just a thin wrapper around
  `DistributedDataParallel`. The other is `FairScaleFsdpAccelerator`, which wraps FairScale's
  [`FullyShardedDataParallel`](https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html).
  You can specify the `DdpAccelerator` in the "distributed" section of a configuration file under the key "ddp_accelerator".
- Added a module `allennlp.nn.checkpoint` with a new base class, `CheckpointWrapper`, for implementations
  of activation/gradient checkpointing. Two implentations are provided. The default implementation is `TorchCheckpointWrapper` (registered as "torch"),
  which exposes [PyTorch's checkpoint functionality](https://pytorch.org/docs/stable/checkpoint.html).
  The other is `FairScaleCheckpointWrapper` which exposes the more flexible
  [checkpointing funtionality from FairScale](https://fairscale.readthedocs.io/en/latest/api/nn/checkpoint/checkpoint_activations.html).
- The `Model` base class now takes a `ddp_accelerator` parameter (an instance of `DdpAccelerator`) which will be available as
  `self.ddp_accelerator` during distributed training. This is useful when, for example, instantiating submodules in your
  model's `__init__()` method by wrapping them with `self.ddp_accelerator.wrap_module()`. See the `allennlp.modules.transformer.t5`
  for an example.
- We now log batch metrics to tensorboard and wandb.
- Added Tango components, to be explored in detail in a later post
- Added `ScaledDotProductMatrixAttention`, and converted the transformer toolkit to use it
- Added tests to ensure that all `Attention` and `MatrixAttention` implementations are interchangeable

### Fixed

- Fixed a bug in `ConditionalRandomField`: `transitions` and `tag_sequence` tensors were not initialized on the desired device causing high CPU usage (see https://github.com/allenai/allennlp/issues/2884)
- Fixed a mispelling: the parameter `contructor_extras` in `Lazy()` is now correctly called `constructor_extras`.
- Fixed broken links in `allennlp.nn.initializers` docs.
- Fixed bug in `BeamSearch` where `last_backpointers` was not being passed to any `Constraint`s.
- `TransformerTextField` can now take tensors of shape `(1, n)` like the tensors produced from a HuggingFace tokenizer.
- `tqdm` lock is now set inside `MultiProcessDataLoading` when new workers are spawned to avoid contention when writing output.
- `ConfigurationError` is now pickleable.
- Multitask models now support `TextFieldTensor` in heads, not just in the backbone.
- Fixed the signature of `ScaledDotProductAttention` to match the other `Attention` classes

### Changed

- The type of the `grad_norm` parameter of `GradientDescentTrainer` is now `Union[float, bool]`,
  with a default value of `False`. `False` means gradients are not rescaled and the gradient
  norm is never even calculated. `True` means the gradients are still not rescaled but the gradient
  norm is calculated and passed on to callbacks. A `float` value means gradients are rescaled.
- `TensorCache` now supports more concurrent readers and writers.
- We no longer log parameter statistics to tensorboard or wandb by default.


## [v2.6.0](https://github.com/allenai/allennlp/releases/tag/v2.6.0) - 2021-07-19

### Added

- Added `on_backward` training callback which allows for control over backpropagation and gradient manipulation.
- Added `AdversarialBiasMitigator`, a Model wrapper to adversarially mitigate biases in predictions produced by a pretrained model for a downstream task.
- Added `which_loss` parameter to `ensure_model_can_train_save_and_load` in `ModelTestCase` to specify which loss to test.
- Added `**kwargs` to `Predictor.from_path()`. These key-word argument will be passed on to the `Predictor`'s constructor.
- The activation layer in the transformer toolkit now can be queried for its output dimension.
- `TransformerEmbeddings` now takes, but ignores, a parameter for the attention mask. This is needed for compatibility with some other modules that get called the same way and use the mask.
- `TransformerPooler` can now be instantiated from a pretrained transformer module, just like the other modules in the transformer toolkit.
- `TransformerTextField`, for cases where you don't care about AllenNLP's advanced text handling capabilities.
- Added `TransformerModule._post_load_pretrained_state_dict_hook()` method. Can be used to modify `missing_keys` and `unexpected_keys` after
  loading a pretrained state dictionary. This is useful when tying weights, for example.
- Added an end-to-end test for the Transformer Toolkit.
- Added `vocab` argument to `BeamSearch`, which is passed to each contraint in `constraints` (if provided).

### Fixed

- Fixed missing device mapping in the `allennlp.modules.conditional_random_field.py` file.
- Fixed Broken link in `allennlp.fairness.fairness_metrics.Separation` docs
- Ensured all `allennlp` submodules are imported with `allennlp.common.plugins.import_plugins()`.
- Fixed `IndexOutOfBoundsException` in `MultiOptimizer` when checking if optimizer received any parameters.
- Removed confusing zero mask from VilBERT.
- Ensured `ensure_model_can_train_save_and_load` is consistently random.
- Fixed weight tying logic in `T5` transformer module. Previously input/output embeddings were always tied. Now this is optional,
  and the default behavior is taken from the `config.tie_word_embeddings` value when instantiating `from_pretrained_module()`.
- Implemented slightly faster label smoothing.
- Fixed the docs for `PytorchTransformerWrapper`
- Fixed recovering training jobs with models that expect `get_metrics()` to not be called until they have seen at least one batch.
- Made the Transformer Toolkit compatible with transformers that don't start their positional embeddings at 0.
- Weights & Biases training callback ("wandb") now works when resuming training jobs.

### Changed

- Changed behavior of `MultiOptimizer` so that while a default optimizer is still required, an error is not thrown if the default optimizer receives no parameters.
- Made the epsilon parameter for the layer normalization in token embeddings configurable. 

### Removed

- Removed `TransformerModule._tied_weights`. Weights should now just be tied directly in the `__init__()` method.
  You can also override `TransformerModule._post_load_pretrained_state_dict_hook()` to remove keys associated with tied weights from `missing_keys`
  after loading a pretrained state dictionary.


## [v2.5.0](https://github.com/allenai/allennlp/releases/tag/v2.5.0) - 2021-06-03

### Added

- Added `TaskSuite` base class and command line functionality for running [`checklist`](https://github.com/marcotcr/checklist) test suites, along with implementations for `SentimentAnalysisSuite`, `QuestionAnsweringSuite`, and `TextualEntailmentSuite`. These can be found in the `allennlp.confidence_checks.task_checklists` module.
- Added `BiasMitigatorApplicator`, which wraps any Model and mitigates biases by finetuning
on a downstream task.
- Added `allennlp diff` command to compute a diff on model checkpoints, analogous to what `git diff` does on two files.
- Meta data defined by the class `allennlp.common.meta.Meta` is now saved in the serialization directory and archive file
  when training models from the command line. This is also now part of the `Archive` named tuple that's returned from `load_archive()`.
- Added `nn.util.distributed_device()` helper function.
- Added `allennlp.nn.util.load_state_dict` helper function.
- Added a way to avoid downloading and loading pretrained weights in modules that wrap transformers
  such as the `PretrainedTransformerEmbedder` and `PretrainedTransformerMismatchedEmbedder`.
  You can do this by setting the parameter `load_weights` to `False`.
  See [PR #5172](https://github.com/allenai/allennlp/pull/5172) for more details.
- Added `SpanExtractorWithSpanWidthEmbedding`, putting specific span embedding computations into the `_embed_spans` method and leaving the common code in `SpanExtractorWithSpanWidthEmbedding` to unify the arguments, and modified `BidirectionalEndpointSpanExtractor`, `EndpointSpanExtractor` and `SelfAttentiveSpanExtractor` accordingly. Now, `SelfAttentiveSpanExtractor` can also embed span widths.
- Added a `min_steps` parameter to `BeamSearch` to set a minimum length for the predicted sequences.
- Added the `FinalSequenceScorer` abstraction to calculate the final scores of the generated sequences in `BeamSearch`. 
- Added `shuffle` argument to `BucketBatchSampler` which allows for disabling shuffling.
- Added `allennlp.modules.transformer.attention_module` which contains a generalized `AttentionModule`. `SelfAttention` and `T5Attention` both inherit from this.
- Added a `Constraint` abstract class to `BeamSearch`, which allows for incorporating constraints on the predictions found by `BeamSearch`,
  along with a `RepeatedNGramBlockingConstraint` constraint implementation, which allows for preventing repeated n-grams in the output from `BeamSearch`.
- Added `DataCollator` for dynamic operations for each batch.

### Changed

- Use `dist_reduce_sum` in distributed metrics.
- Allow Google Cloud Storage paths in `cached_path` ("gs://...").
- Renamed `nn.util.load_state_dict()` to `read_state_dict` to avoid confusion with `torch.nn.Module.load_state_dict()`.
- `TransformerModule.from_pretrained_module` now only accepts a pretrained model ID (e.g. "bert-base-case") instead of
  an actual `torch.nn.Module`. Other parameters to this method have changed as well.
- Print the first batch to the console by default.
- Renamed `sanity_checks` to `confidence_checks` (`sanity_checks` is deprecated and will be removed in AllenNLP 3.0).
- Trainer callbacks can now store and restore state in case a training run gets interrupted.
- VilBERT backbone now rolls and unrolls extra dimensions to handle input with > 3 dimensions.
- `BeamSearch` is now a `Registrable` class.

### Fixed

- When `PretrainedTransformerIndexer` folds long sequences, it no longer loses the information from token type ids.
- Fixed documentation for `GradientDescentTrainer.cuda_device`.
- Re-starting a training run from a checkpoint in the middle of an epoch now works correctly.
- When using the "moving average" weights smoothing feature of the trainer, training checkpoints would also get smoothed, with strange results for resuming a training job. This has been fixed.
- When re-starting an interrupted training job, the trainer will now read out the data loader even for epochs and batches that can be skipped. We do this to try to get any random number generators used by the reader or data loader into the same state as they were the first time the training job ran.
- Fixed the potential for a race condition with `cached_path()` when extracting archives. Although the race condition
  is still possible if used with `force_extract=True`.
- Fixed `wandb` callback to work in distributed training.
- Fixed `tqdm` logging into multiple files with `allennlp-optuna`.


## [v2.4.0](https://github.com/allenai/allennlp/releases/tag/v2.4.0) - 2021-04-22

### Added

- Added a T5 implementation to `modules.transformers`.

### Changed

- Weights & Biases callback can now work in anonymous mode (i.e. without the `WANDB_API_KEY` environment variable).

### Fixed

- The `GradientDescentTrainer` no longer leaves stray model checkpoints around when it runs out of patience.
- Fixed `cached_path()` for "hf://" files.
- Improved the error message for the `PolynomialDecay` LR scheduler when `num_steps_per_epoch` is missing.


## [v2.3.1](https://github.com/allenai/allennlp/releases/tag/v2.3.1) - 2021-04-20

### Added

- Added support for the HuggingFace Hub as an alternative way to handle loading files. Hub downloads should be made through the `hf://` URL scheme.
- Add new dimension to the `interpret` module: influence functions via the `InfluenceInterpreter` base class, along with a concrete implementation: `SimpleInfluence`.
- Added a `quiet` parameter to the `MultiProcessDataLoading` that disables `Tqdm` progress bars.
- The test for distributed metrics now takes a parameter specifying how often you want to run it.
- Created the fairness module and added three fairness metrics: `Independence`, `Separation`, and `Sufficiency`.
- Added four bias metrics to the fairness module: `WordEmbeddingAssociationTest`, `EmbeddingCoherenceTest`, `NaturalLanguageInference`, and `AssociationWithoutGroundTruth`.
- Added four bias direction methods (`PCABiasDirection`, `PairedPCABiasDirection`, `TwoMeansBiasDirection`, `ClassificationNormalBiasDirection`) and four bias mitigation methods (`LinearBiasMitigator`, `HardBiasMitigator`, `INLPBiasMitigator`, `OSCaRBiasMitigator`).

### Changed

- Updated CONTRIBUTING.md to remind reader to upgrade pip setuptools to avoid spaCy installation issues.

### Fixed

- Fixed a bug with the `ShardedDatasetReader` when used with multi-process data loading (https://github.com/allenai/allennlp/issues/5132).

## [v2.3.0](https://github.com/allenai/allennlp/releases/tag/v2.3.0) - 2021-04-14

### Added

- Ported the following Huggingface `LambdaLR`-based schedulers: `ConstantLearningRateScheduler`, `ConstantWithWarmupLearningRateScheduler`, `CosineWithWarmupLearningRateScheduler`, `CosineHardRestartsWithWarmupLearningRateScheduler`.
- Added new `sub_token_mode` parameter to `pretrained_transformer_mismatched_embedder` class to support first sub-token embedding
- Added a way to run a multi task model with a dataset reader as part of `allennlp predict`.
- Added new `eval_mode` in `PretrainedTransformerEmbedder`. If it is set to `True`, the transformer is _always_ run in evaluation mode, which, e.g., disables dropout and does not update batch normalization statistics.
- Added additional parameters to the W&B callback: `entity`, `group`, `name`, `notes`, and `wandb_kwargs`.

### Changed

- Sanity checks in the `GradientDescentTrainer` can now be turned off by setting the `run_sanity_checks` parameter to `False`.
- Allow the order of examples in the task cards to be specified explicitly
- `histogram_interval` parameter is now deprecated in `TensorboardWriter`, please use `distribution_interval` instead.
- Memory usage is not logged in tensorboard during training now. `ConsoleLoggerCallback` should be used instead.
- If you use the `min_count` parameter of the Vocabulary, but you specify a namespace that does not exist, the vocabulary creation will raise a `ConfigurationError`.
- Documentation updates made to SoftmaxLoss regarding padding and the expected shapes of the input and output tensors of `forward`.
- Moved the data preparation script for coref into allennlp-models.
- If a transformer is not in cache but has override weights, the transformer's pretrained weights are no longer downloaded, that is, only its `config.json` file is downloaded.
- `SanityChecksCallback` now raises `SanityCheckError` instead of `AssertionError` when a check fails.
- `jsonpickle` removed from dependencies.
- Improved the error message from `Registrable.by_name()` when the name passed does not match any registered subclassess.
  The error message will include a suggestion if there is a close match between the name passed and a registered name.

### Fixed

- Fixed a bug where some `Activation` implementations could not be pickled due to involving a lambda function.
- Fixed `__str__()` method on `ModelCardInfo` class.
- Fixed a stall when using distributed training and gradient accumulation at the same time
- Fixed an issue where using the `from_pretrained_transformer` `Vocabulary` constructor in distributed training via the `allennlp train` command
  would result in the data being iterated through unnecessarily.
- Fixed a bug regarding token indexers with the `InterleavingDatasetReader` when used with multi-process data loading.
- Fixed a warning from `transformers` when using `max_length` in the `PretrainedTransformerTokenizer`.

### Removed

- Removed the `stride` parameter to `PretrainedTransformerTokenizer`. This parameter had no effect.


## [v2.2.0](https://github.com/allenai/allennlp/releases/tag/v2.2.0) - 2021-03-26


### Added

- Add new method on `Field` class: `.human_readable_repr() -> Any`
- Add new method on `Instance` class: `.human_readable_dict() -> JsonDict`.
- Added `WandBCallback` class for [Weights & Biases](https://wandb.ai) integration, registered as a callback under
  the name "wandb".
- Added `TensorBoardCallback` to replace the `TensorBoardWriter`. Registered as a callback
  under the name "tensorboard".
- Added `NormalizationBiasVerification` and `SanityChecksCallback` for model sanity checks.
- `SanityChecksCallback` runs by default from the `allennlp train` command.
  It can be turned off by setting `trainer.enable_default_callbacks` to `false` in your config.

### Changed

- Use attributes of `ModelOutputs` object in `PretrainedTransformerEmbedder` instead of indexing.
- Added support for PyTorch version 1.8 and `torchvision` version 0.9 .
- `Model.get_parameters_for_histogram_tensorboard_logging` is deprecated in favor of
  `Model.get_parameters_for_histogram_logging`.


### Fixed

- Makes sure tensors that are stored in `TensorCache` always live on CPUs
- Fixed a bug where `FromParams` objects wrapped in `Lazy()` couldn't be pickled.
- Fixed a bug where the `ROUGE` metric couldn't be picked.
- Fixed a bug reported by https://github.com/allenai/allennlp/issues/5036. We keeps our spacy POS tagger on.

### Removed

- Removed `TensorBoardWriter`. Please use the `TensorBoardCallback` instead.


## [v2.1.0](https://github.com/allenai/allennlp/releases/tag/v2.1.0) - 2021-02-24

### Changed

- `coding_scheme` parameter is now deprecated in `Conll2003DatasetReader`, please use `convert_to_coding_scheme` instead.
- Support spaCy v3

### Added

- Added `ModelUsage` to `ModelCard` class.
- Added a way to specify extra parameters to the predictor in an `allennlp predict` call.
- Added a way to initialize a `Vocabulary` from transformers models.
- Added the ability to use `Predictors` with multitask models through the new `MultiTaskPredictor`.
- Added an example for fields of type `ListField[TextField]` to `apply_token_indexers` API docs.
- Added `text_key` and `label_key` parameters to `TextClassificationJsonReader` class.
- Added `MultiOptimizer`, which allows you to use different optimizers for different parts of your model.
- Added a clarification to `predictions_to_labeled_instances` API docs for attack from json

### Fixed

- `@Registrable.register(...)` decorator no longer masks the decorated class's annotations
- Ensured that `MeanAbsoluteError` always returns a `float` metric value instead of a `Tensor`.
- Learning rate schedulers that rely on metrics from the validation set were broken in v2.0.0. This
  brings that functionality back.
- Fixed a bug where the `MultiProcessDataLoading` would crash when `num_workers > 0`, `start_method = "spawn"`, `max_instances_in_memory not None`, and `batches_per_epoch not None`.
- Fixed documentation and validation checks for `FBetaMultiLabelMetric`.
- Fixed handling of HTTP errors when fetching remote resources with `cached_path()`. Previously the content would be cached even when
  certain errors - like 404s - occurred. Now an `HTTPError` will be raised whenever the HTTP response is not OK.
- Fixed a bug where the `MultiTaskDataLoader` would crash when `num_workers > 0`
- Fixed an import error that happens when PyTorch's distributed framework is unavailable on the system.


## [v2.0.1](https://github.com/allenai/allennlp/releases/tag/v2.0.1) - 2021-01-29

### Added

- Added `tokenizer_kwargs` and `transformer_kwargs` arguments to `PretrainedTransformerBackbone`
- Resize transformers word embeddings layer for `additional_special_tokens`

### Changed

- GradientDescentTrainer makes `serialization_dir` when it's instantiated, if it doesn't exist.

### Fixed

- `common.util.sanitize` now handles sets.


## [v2.0.0](https://github.com/allenai/allennlp/releases/tag/v2.0.0) - 2021-01-27

### Added

- The `TrainerCallback` constructor accepts `serialization_dir` provided by `Trainer`. This can be useful for `Logger` callbacks those need to store files in the run directory.
- The `TrainerCallback.on_start()` is fired at the start of the training.
- The `TrainerCallback` event methods now accept `**kwargs`. This may be useful to maintain backwards-compability of callbacks easier in the future. E.g. we may decide to pass the exception/traceback object in case of failure to `on_end()` and this older callbacks may simply ignore the argument instead of raising a `TypeError`.
- Added a `TensorBoardCallback` which wraps the `TensorBoardWriter`.

### Changed

- The `TrainerCallack.on_epoch()` does not fire with `epoch=-1` at the start of the training.
  Instead, `TrainerCallback.on_start()` should be used for these cases.
- `TensorBoardBatchMemoryUsage` is converted from `BatchCallback` into `TrainerCallback`.
- `TrackEpochCallback` is converted from `EpochCallback` into `TrainerCallback`.
- `Trainer` can accept callbacks simply with name `callbacks` instead of `trainer_callbacks`.
- `TensorboardWriter` renamed to `TensorBoardWriter`, and removed as an argument to the `GradientDescentTrainer`.
  In order to enable TensorBoard logging during training, you should utilize the `TensorBoardCallback` instead.

### Removed

- Removed `EpochCallback`, `BatchCallback` in favour of `TrainerCallback`.
  The metaclass-wrapping implementation is removed as well.
- Removed the `tensorboard_writer` parameter to `GradientDescentTrainer`. You should use the `TensorBoardCallback` now instead.

### Fixed

- Now Trainer always fires `TrainerCallback.on_end()` so all the resources can be cleaned up properly.
- Fixed the misspelling, changed `TensoboardBatchMemoryUsage` to `TensorBoardBatchMemoryUsage`.
- We set a value to `epoch` so in case of firing `TrainerCallback.on_end()` the variable is bound.
  This could have lead to an error in case of trying to recover a run after it was finished training.


## [v2.0.0rc1](https://github.com/allenai/allennlp/releases/tag/v2.0.0rc1) - 2021-01-21

### Added

- Added `TensorCache` class for caching tensors on disk
- Added abstraction and concrete implementation for image loading
- Added abstraction and concrete implementation for `GridEmbedder`
- Added abstraction and demo implementation for an image augmentation module.
- Added abstraction and concrete implementation for region detectors.
- A new high-performance default `DataLoader`: `MultiProcessDataLoading`.
- A `MultiTaskModel` and abstractions to use with it, including `Backbone` and `Head`.  The
  `MultiTaskModel` first runs its inputs through the `Backbone`, then passes the result (and
  whatever other relevant inputs it got) to each `Head` that's in use.
- A `MultiTaskDataLoader`, with a corresponding `MultiTaskDatasetReader`, and a couple of new
  configuration objects: `MultiTaskEpochSampler` (for deciding what proportion to sample from each
  dataset at every epoch) and a `MultiTaskScheduler` (for ordering the instances within an epoch).
- Transformer toolkit to plug and play with modular components of transformer architectures.
- Added a command to count the number of instances we're going to be training with
- Added a `FileLock` class to `common.file_utils`. This is just like the `FileLock` from the `filelock` library, except that
  it adds an optional flag `read_only_ok: bool`, which when set to `True` changes the behavior so that a warning will be emitted
  instead of an exception when lacking write permissions on an existing file lock.
  This makes it possible to use the `FileLock` class on a read-only file system.
- Added a new learning rate scheduler: `CombinedLearningRateScheduler`. This can be used to combine different LR schedulers, using one after the other.
- Added an official CUDA 10.1 Docker image.
- Moving `ModelCard` and `TaskCard` abstractions into the main repository.
- Added a util function `allennlp.nn.util.dist_reduce(...)` for handling distributed reductions.
  This is especially useful when implementing a distributed `Metric`.
- Added a `FileLock` class to `common.file_utils`. This is just like the `FileLock` from the `filelock` library, except that
  it adds an optional flag `read_only_ok: bool`, which when set to `True` changes the behavior so that a warning will be emitted
  instead of an exception when lacking write permissions on an existing file lock.
  This makes it possible to use the `FileLock` class on a read-only file system.
- Added a new learning rate scheduler: `CombinedLearningRateScheduler`. This can be used to combine different LR schedulers, using one after the other.
- Moving `ModelCard` and `TaskCard` abstractions into the main repository.

### Changed

- `DatasetReader`s are now always lazy. This means there is no `lazy` parameter in the base
  class, and the `_read()` method should always be a generator.
- The `DataLoader` now decides whether to load instances lazily or not.
  With the `PyTorchDataLoader` this is controlled with the `lazy` parameter, but with
  the `MultiProcessDataLoading` this is controlled by the `max_instances_in_memory` setting.
- `ArrayField` is now called `TensorField`, and implemented in terms of torch tensors, not numpy.
- Improved `nn.util.move_to_device` function by avoiding an unnecessary recursive check for tensors and
  adding a `non_blocking` optional argument, which is the same argument as in `torch.Tensor.to()`.
- If you are trying to create a heterogeneous batch, you now get a better error message.
- Readers using the new vision features now explicitly log how they are featurizing images.
- `master_addr` and `master_port` renamed to `primary_addr` and `primary_port`, respectively.
- `is_master` parameter for training callbacks renamed to `is_primary`.
- `master` branch renamed to `main`
- Torch version bumped to 1.7.1 in Docker images.
- 'master' branch renamed to 'main'
- Torch version bumped to 1.7.1 in Docker images.

### Removed

- Removed `nn.util.has_tensor`.

### Fixed

- The `build-vocab` command no longer crashes when the resulting vocab file is
  in the current working directory.
- VQA models now use the `vqa_score` metric for early stopping. This results in
  much better scores.
- Fixed typo with `LabelField` string representation: removed trailing apostrophe.
- `Vocabulary.from_files` and `cached_path` will issue a warning, instead of failing, when a lock on an existing resource
  can't be acquired because the file system is read-only.
- `TrackEpochCallback` is now a `EpochCallback`.


## [v1.3.0](https://github.com/allenai/allennlp/releases/tag/v1.3.0) - 2020-12-15

### Added

- Added links to source code in docs.
- Added `get_embedding_layer` and `get_text_field_embedder` to the `Predictor` class; to specify embedding layers for non-AllenNLP models.
- Added [Gaussian Error Linear Unit (GELU)](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html) as an Activation.

### Changed

- Renamed module `allennlp.data.tokenizers.token` to `allennlp.data.tokenizers.token_class` to avoid
  [this bug](https://github.com/allenai/allennlp/issues/4819).
- `transformers` dependency updated to version 4.0.1.
- `BasicClassifier`'s forward method now takes a metadata field.

### Fixed

- Fixed a lot of instances where tensors were first created and then sent to a device
  with `.to(device)`. Instead, these tensors are now created directly on the target device.
- Fixed issue with `GradientDescentTrainer` when constructed with `validation_data_loader=None` and `learning_rate_scheduler!=None`.
- Fixed a bug when removing all handlers in root logger.
- `ShardedDatasetReader` now inherits parameters from `base_reader` when required.
- Fixed an issue in `FromParams` where parameters in the `params` object used to a construct a class
  were not passed to the constructor if the value of the parameter was equal to the default value.
  This caused bugs in some edge cases where a subclass that takes `**kwargs` needs to inspect
  `kwargs` before passing them to its superclass.
- Improved the band-aid solution for segmentation faults and the "ImportError: dlopen: cannot load any more object with static TLS"
  by adding a `transformers` import.
- Added safety checks for extracting tar files
- Turned superfluous warning to info when extending the vocab in the embedding matrix, if no pretrained file was provided


## [v1.2.2](https://github.com/allenai/allennlp/releases/tag/v1.2.2) - 2020-11-17

### Added

- Added Docker builds for other torch-supported versions of CUDA.
- Adds [`allennlp-semparse`](https://github.com/allenai/allennlp-semparse) as an official, default plugin.

### Fixed

- `GumbelSampler` now sorts the beams by their true log prob.


## [v1.2.1](https://github.com/allenai/allennlp/releases/tag/v1.2.1) - 2020-11-10

### Added

- Added an optional `seed` parameter to `ModelTestCase.set_up_model` which sets the random
  seed for `random`, `numpy`, and `torch`.
- Added support for a global plugins file at `~/.allennlp/plugins`.
- Added more documentation about plugins.
- Added sampler class and parameter in beam search for non-deterministic search, with several
  implementations, including `MultinomialSampler`, `TopKSampler`, `TopPSampler`, and
  `GumbelSampler`. Utilizing `GumbelSampler` will give [Stochastic Beam Search](https://api.semanticscholar.org/CorpusID:76662039).

### Changed

- Pass batch metrics to `BatchCallback`.

### Fixed

- Fixed a bug where forward hooks were not cleaned up with saliency interpreters if there
  was an exception.
- Fixed the computation of saliency maps in the Interpret code when using mismatched indexing.
  Previously, we would compute gradients from the top of the transformer, after aggregation from
  wordpieces to tokens, which gives results that are not very informative.  Now, we compute gradients
  with respect to the embedding layer, and aggregate wordpieces to tokens separately.
- Fixed the heuristics for finding embedding layers in the case of RoBERTa. An update in the
  `transformers` library broke our old heuristic.
- Fixed typo with registered name of ROUGE metric. Previously was `rogue`, fixed to `rouge`.
- Fixed default masks that were erroneously created on the CPU even when a GPU is available.
- Fixed pretrained embeddings for transformers that don't use end tokens.
- Fixed the transformer tokenizer cache when the tokenizers are initialized with custom kwargs.


## [v1.2.0](https://github.com/allenai/allennlp/releases/tag/v1.2.0) - 2020-10-29

### Changed

- Enforced stricter typing requirements around the use of `Optional[T]` types.
- Changed the behavior of `Lazy` types in `from_params` methods. Previously, if you defined a `Lazy` parameter like
  `foo: Lazy[Foo] = None` in a custom `from_params` classmethod, then `foo` would actually never be `None`.
  This behavior is now different. If no params were given for `foo`, it will be `None`.
  You can also now set default values for foo like `foo: Lazy[Foo] = Lazy(Foo)`.
  Or, if you want you want a default value but also want to allow for `None` values, you can
  write it like this: `foo: Optional[Lazy[Foo]] = Lazy(Foo)`.
- Added support for PyTorch version 1.7.

### Fixed

- Made it possible to instantiate `TrainerCallback` from config files.
- Fixed the remaining broken internal links in the API docs.
- Fixed a bug where Hotflip would crash with a model that had multiple TokenIndexers and the input
  used rare vocabulary items.
- Fixed a bug where `BeamSearch` would fail if `max_steps` was equal to 1.
- Fixed `BasicTextFieldEmbedder` to not raise ConfigurationError if it has embedders that are empty and not in input


## [v1.2.0rc1](https://github.com/allenai/allennlp/releases/tag/v1.2.0rc1) - 2020-10-22

### Added

- Added a warning when `batches_per_epoch` for the validation data loader is inherited from
  the train data loader.
- Added a `build-vocab` subcommand that can be used to build a vocabulary from a training config file.
- Added `tokenizer_kwargs` argument to `PretrainedTransformerMismatchedIndexer`.
- Added `tokenizer_kwargs` and `transformer_kwargs` arguments to `PretrainedTransformerMismatchedEmbedder`.
- Added official support for Python 3.8.
- Added a script: `scripts/release_notes.py`, which automatically prepares markdown release notes from the
  CHANGELOG and commit history.
- Added a flag `--predictions-output-file` to the `evaluate` command, which tells AllenNLP to write the
  predictions from the given dataset to the file as JSON lines.
- Added the ability to ignore certain missing keys when loading a model from an archive. This is done
  by adding a class-level variable called `authorized_missing_keys` to any PyTorch module that a `Model` uses.
  If defined, `authorized_missing_keys` should be a list of regex string patterns.
- Added `FBetaMultiLabelMeasure`, a multi-label Fbeta metric. This is a subclass of the existing `FBetaMeasure`.
- Added ability to pass additional key word arguments to `cached_transformers.get()`, which will be passed on to `AutoModel.from_pretrained()`.
- Added an `overrides` argument to `Predictor.from_path()`.
- Added a `cached-path` command.
- Added a function `inspect_cache` to `common.file_utils` that prints useful information about the cache. This can also
  be used from the `cached-path` command with `allennlp cached-path --inspect`.
- Added a function `remove_cache_entries` to `common.file_utils` that removes any cache entries matching the given
  glob patterns. This can used from the `cached-path` command with `allennlp cached-path --remove some-files-*`.
- Added logging for the main process when running in distributed mode.
- Added a `TrainerCallback` object to support state sharing between batch and epoch-level training callbacks.
- Added support for .tar.gz in PretrainedModelInitializer.
- Made `BeamSearch` instantiable `from_params`.
- Pass `serialization_dir` to `Model` and `DatasetReader`.
- Added an optional `include_in_archive` parameter to the top-level of configuration files. When specified, `include_in_archive` should be a list of paths relative to the serialization directory which will be bundled up with the final archived model from a training run.

### Changed

- Subcommands that don't require plugins will no longer cause plugins to be loaded or have an `--include-package` flag.
- Allow overrides to be JSON string or `dict`.
- `transformers` dependency updated to version 3.1.0.
- When `cached_path` is called on a local archive with `extract_archive=True`, the archive is now extracted into a unique subdirectory of the cache root instead of a subdirectory of the archive's directory. The extraction directory is also unique to the modification time of the archive, so if the file changes, subsequent calls to `cached_path` will know to re-extract the archive.
- Removed the `truncation_strategy` parameter to `PretrainedTransformerTokenizer`. The way we're calling the tokenizer, the truncation strategy takes no effect anyways.
- Don't use initializers when loading a model, as it is not needed.
- Distributed training will now automatically search for a local open port if the `master_port` parameter is not provided.
- In training, save model weights before evaluation.
- `allennlp.common.util.peak_memory_mb` renamed to `peak_cpu_memory`, and `allennlp.common.util.gpu_memory_mb` renamed to `peak_gpu_memory`,
  and they both now return the results in bytes as integers. Also, the `peak_gpu_memory` function now utilizes PyTorch functions to find the memory
  usage instead of shelling out to the `nvidia-smi` command. This is more efficient and also more accurate because it only takes
  into account the tensor allocations of the current PyTorch process.
- Make sure weights are first loaded to the cpu when using PretrainedModelInitializer, preventing wasted GPU memory.
- Load dataset readers in `load_archive`.
- Updated `AllenNlpTestCase` docstring to remove reference to `unittest.TestCase`

### Removed

- Removed `common.util.is_master` function.

### Fixed

- Fix CUDA/CPU device mismatch bug during distributed training for categorical accuracy metric.
- Fixed a bug where the reported `batch_loss` metric was incorrect when training with gradient accumulation.
- Class decorators now displayed in API docs.
- Fixed up the documentation for the `allennlp.nn.beam_search` module.
- Ignore `*args` when constructing classes with `FromParams`.
- Ensured some consistency in the types of the values that metrics return.
- Fix a PyTorch warning by explicitly providing the `as_tuple` argument (leaving
  it as its default value of `False`) to `Tensor.nonzero()`.
- Remove temporary directory when extracting model archive in `load_archive`
  at end of function rather than via `atexit`.
- Fixed a bug where using `cached_path()` offline could return a cached resource's lock file instead
  of the cache file.
- Fixed a bug where `cached_path()` would fail if passed a `cache_dir` with the user home shortcut `~/`.
- Fixed a bug in our doc building script where markdown links did not render properly
  if the "href" part of the link (the part inside the `()`) was on a new line.
- Changed how gradients are zeroed out with an optimization. See [this video from NVIDIA](https://www.youtube.com/watch?v=9mS1fIYj1So)
  at around the 9 minute mark.
- Fixed a bug where parameters to a `FromParams` class that are dictionaries wouldn't get logged
  when an instance is instantiated `from_params`.
- Fixed a bug in distributed training where the vocab would be saved from every worker, when it should have been saved by only the local master process.
- Fixed a bug in the calculation of rouge metrics during distributed training where the total sequence count was not being aggregated across GPUs.
- Fixed `allennlp.nn.util.add_sentence_boundary_token_ids()` to use `device` parameter of input tensor.
- Be sure to close the TensorBoard writer even when training doesn't finish.
- Fixed the docstring for `PyTorchSeq2VecWrapper`.
- Fixed a bug in the cnn_encoder where activations involving masked tokens could be picked up by the max
- Fix intra word tokenization for `PretrainedTransformerTokenizer` when disabling fast tokenizer.


## [v1.1.0](https://github.com/allenai/allennlp/releases/tag/v1.1.0) - 2020-09-08

### Fixed

- Fixed handling of some edge cases when constructing classes with `FromParams` where the class
  accepts `**kwargs`.
- Fixed division by zero error when there are zero-length spans in the input to a
  `PretrainedTransformerMismatchedIndexer`.
- Improved robustness of `cached_path` when extracting archives so that the cache won't be corrupted
  if a failure occurs during extraction.
- Fixed a bug with the `average` and `evalb_bracketing_score` metrics in distributed training.

### Added

- `Predictor.capture_model_internals()` now accepts a regex specifying which modules to capture.


## [v1.1.0rc4](https://github.com/allenai/allennlp/releases/tag/v1.1.0rc4) - 2020-08-20

### Added

- Added a workflow to GitHub Actions that will automatically close unassigned stale issues and
  ping the assignees of assigned stale issues.

### Fixed

- Fixed a bug in distributed metrics that caused nan values due to repeated addition of an accumulated variable.

## [v1.1.0rc3](https://github.com/allenai/allennlp/releases/tag/v1.1.0rc3) - 2020-08-12

### Fixed

- Fixed how truncation was handled with `PretrainedTransformerTokenizer`.
  Previously, if `max_length` was set to `None`, the tokenizer would still do truncation if the
  transformer model had a default max length in its config.
  Also, when `max_length` was set to a non-`None` value, several warnings would appear
  for certain transformer models around the use of the `truncation` parameter.
- Fixed evaluation of all metrics when using distributed training.
- Added a `py.typed` marker. Fixed type annotations in `allennlp.training.util`.
- Fixed problem with automatically detecting whether tokenization is necessary.
  This affected primarily the Roberta SST model.
- Improved help text for using the --overrides command line flag.


## [v1.1.0rc2](https://github.com/allenai/allennlp/releases/tag/v1.1.0rc2) - 2020-07-31

### Changed

- Upgraded PyTorch requirement to 1.6.
- Replaced the NVIDIA Apex AMP module with torch's native AMP module. The default trainer (`GradientDescentTrainer`)
  now takes a `use_amp: bool` parameter instead of the old `opt_level: str` parameter.

### Fixed

- Removed unnecessary warning about deadlocks in `DataLoader`.
- Fixed testing models that only return a loss when they are in training mode.
- Fixed a bug in `FromParams` that caused silent failure in case of the parameter type being `Optional[Union[...]]`.
- Fixed a bug where the program crashes if `evaluation_data_loader` is a `AllennlpLazyDataset`.

### Added

- Added the option to specify `requires_grad: false` within an optimizer's parameter groups.
- Added the `file-friendly-logging` flag back to the `train` command. Also added this flag to the `predict`, `evaluate`, and `find-learning-rate` commands.
- Added an `EpochCallback` to track current epoch as a model class member.
- Added the option to enable or disable gradient checkpointing for transformer token embedders via boolean parameter `gradient_checkpointing`.

### Removed

- Removed the `opt_level` parameter to `Model.load` and `load_archive`. In order to use AMP with a loaded
  model now, just run the model's forward pass within torch's [`autocast`](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast)
  context.

## [v1.1.0rc1](https://github.com/allenai/allennlp/releases/tag/v1.1.0rc1) - 2020-07-14

### Fixed

- Reduced the amount of log messages produced by `allennlp.common.file_utils`.
- Fixed a bug where `PretrainedTransformerEmbedder` parameters appeared to be trainable
  in the log output even when `train_parameters` was set to `False`.
- Fixed a bug with the sharded dataset reader where it would only read a fraction of the instances
  in distributed training.
- Fixed checking equality of `TensorField`s.
- Fixed a bug where `NamespaceSwappingField` did not work correctly with `.empty_field()`.
- Put more sensible defaults on the `huggingface_adamw` optimizer.
- Simplified logging so that all logging output always goes to one file.
- Fixed interaction with the python command line debugger.
- Log the grad norm properly even when we're not clipping it.
- Fixed a bug where `PretrainedModelInitializer` fails to initialize a model with a 0-dim tensor
- Fixed a bug with the layer unfreezing schedule of the `SlantedTriangular` learning rate scheduler.
- Fixed a regression with logging in the distributed setting. Only the main worker should write log output to the terminal.
- Pinned the version of boto3 for package managers (e.g. poetry).
- Fixed issue #4330 by updating the `tokenizers` dependency.
- Fixed a bug in `TextClassificationPredictor` so that it passes tokenized inputs to the `DatasetReader`
  in case it does not have a tokenizer.
- `reg_loss` is only now returned for models that have some regularization penalty configured.
- Fixed a bug that prevented `cached_path` from downloading assets from GitHub releases.
- Fixed a bug that erroneously increased last label's false positive count in calculating fbeta metrics.
- `Tqdm` output now looks much better when the output is being piped or redirected.
- Small improvements to how the API documentation is rendered.
- Only show validation progress bar from main process in distributed training.

### Added

- Adjust beam search to support multi-layer decoder.
- A method to ModelTestCase for running basic model tests when you aren't using config files.
- Added some convenience methods for reading files.
- Added an option to `file_utils.cached_path` to automatically extract archives.
- Added the ability to pass an archive file instead of a local directory to `Vocab.from_files`.
- Added the ability to pass an archive file instead of a glob to `ShardedDatasetReader`.
- Added a new `"linear_with_warmup"` learning rate scheduler.
- Added a check in `ShardedDatasetReader` that ensures the base reader doesn't implement manual
  distributed sharding itself.
- Added an option to `PretrainedTransformerEmbedder` and `PretrainedTransformerMismatchedEmbedder` to use a
  scalar mix of all hidden layers from the transformer model instead of just the last layer. To utilize
  this, just set `last_layer_only` to `False`.
- `cached_path()` can now read files inside of archives.
- Training metrics now include `batch_loss` and `batch_reg_loss` in addition to aggregate loss across number of batches.

### Changed

- Not specifying a `cuda_device` now automatically determines whether to use a GPU or not.
- Discovered plugins are logged so you can see what was loaded.
- `allennlp.data.DataLoader` is now an abstract registrable class. The default implementation
remains the same, but was renamed to `allennlp.data.PyTorchDataLoader`.
- `BertPooler` can now unwrap and re-wrap extra dimensions if necessary.
- New `transformers` dependency. Only version >=3.0 now supported.

## [v1.0.0](https://github.com/allenai/allennlp/releases/tag/v1.0.0) - 2020-06-16

### Fixed

- Lazy dataset readers now work correctly with multi-process data loading.
- Fixed race conditions that could occur when using a dataset cache.

### Added

- A bug where where all datasets would be loaded for vocab creation even if not needed.
- A parameter to the `DatasetReader` class: `manual_multi_process_sharding`. This is similar
  to the `manual_distributed_sharding` parameter, but applies when using a multi-process
  `DataLoader`.

## [v1.0.0rc6](https://github.com/allenai/allennlp/releases/tag/v1.0.0rc6) - 2020-06-11

### Fixed

- A bug where `TextField`s could not be duplicated since some tokenizers cannot be deep-copied.
  See https://github.com/allenai/allennlp/issues/4270.
- Our caching mechanism had the potential to introduce race conditions if multiple processes
  were attempting to cache the same file at once. This was fixed by using a lock file tied to each
  cached file.
- `get_text_field_mask()` now supports padding indices that are not `0`.
- A bug where `predictor.get_gradients()` would return an empty dictionary if an embedding layer had trainable set to false
- Fixes `PretrainedTransformerMismatchedIndexer` in the case where a token consists of zero word pieces.
- Fixes a bug when using a lazy dataset reader that results in a `UserWarning` from PyTorch being printed at
  every iteration during training.
- Predictor names were inconsistently switching between dashes and underscores. Now they all use underscores.
- `Predictor.from_path` now automatically loads plugins (unless you specify `load_plugins=False`) so
  that you don't have to manually import a bunch of modules when instantiating predictors from
  an archive path.
- `allennlp-server` automatically found as a plugin once again.

### Added

- A `duplicate()` method on `Instance`s and `Field`s, to be used instead of `copy.deepcopy()`
- A batch sampler that makes sure each batch contains approximately the same number of tokens (`MaxTokensBatchSampler`)
- Functions to turn a sequence of token indices back into tokens
- The ability to use Huggingface encoder/decoder models as token embedders
- Improvements to beam search
- ROUGE metric
- Polynomial decay learning rate scheduler
- A `BatchCallback` for logging CPU and GPU memory usage to tensorboard. This is mainly for debugging
  because using it can cause a significant slowdown in training.
- Ability to run pretrained transformers as an embedder without training the weights
- Add Optuna Integrated badge to README.md

### Changed

- Similar to our caching mechanism, we introduced a lock file to the vocab to avoid race
  conditions when saving/loading the vocab from/to the same serialization directory in different processes.
- Changed the `Token`, `Instance`, and `Batch` classes along with all `Field` classes to "slots" classes. This dramatically reduces the size in memory of instances.
- SimpleTagger will no longer calculate span-based F1 metric when `calculate_span_f1` is `False`.
- CPU memory for every worker is now reported in the logs and the metrics. Previously this was only reporting the CPU memory of the master process, and so it was only
  correct in the non-distributed setting.
- To be consistent with PyTorch `IterableDataset`, `AllennlpLazyDataset` no longer implements `__len__()`.
  Previously it would always return 1.
- Removed old tutorials, in favor of [the new AllenNLP Guide](https://guide.allennlp.org)
- Changed the vocabulary loading to consider new lines for Windows/Linux and Mac.

## [v1.0.0rc5](https://github.com/allenai/allennlp/releases/tag/v1.0.0rc5) - 2020-05-26

### Fixed

- Fix bug where `PretrainedTransformerTokenizer` crashed with some transformers (#4267)
- Make `cached_path` work offline.
- Tons of docstring inconsistencies resolved.
- Nightly builds no longer run on forks.
- Distributed training now automatically figures out which worker should see which instances
- A race condition bug in distributed training caused from saving the vocab to file from the master process while other processing might be reading those files.
- Unused dependencies in `setup.py` removed.

### Added

- Additional CI checks to ensure docstrings are consistently formatted.
- Ability to train on CPU with multiple processes by setting `cuda_devices` to a list of negative integers in your training config. For example: `"distributed": {"cuda_devices": [-1, -1]}`. This is mainly to make it easier to test and debug distributed training code..
- Documentation for when parameters don't need config file entries.

### Changed

- The `allennlp test-install` command now just ensures the core submodules can
be imported successfully, and prints out some other useful information such as the version, PyTorch version,
and the number of GPU devices available.
- All of the tests moved from `allennlp/tests` to `tests` at the root level, and
`allennlp/tests/fixtures` moved to `test_fixtures` at the root level. The PyPI source and wheel distributions will no longer include tests and fixtures.

## [v1.0.0rc4](https://github.com/allenai/allennlp/releases/tag/v1.0.0rc4) - 2020-05-14

We first introduced this `CHANGELOG` after release `v1.0.0rc4`, so please refer to the GitHub release
notes for this and earlier releases.
