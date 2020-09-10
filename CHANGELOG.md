# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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

- `Predictor.capture_model_internals()` now accepts a regex specifying
  which modules to capture


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
- Fixed checking equality of `ArrayField`s.
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
