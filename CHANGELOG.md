# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Reduced the amount of log messages produced by `allennlp.common.file_utils`.
- Fixed a bug where `PretrainedTransformerEmbedder` parameters appeared to be trainable
  in the log output even when `train_parameters` was set to `False`.
- Fixed a bug with the sharded dataset reader where it would only read a fraction of the instances
  in distributed training.
- Fixed checking equality of `ArrayField`s.
- Fixed a bug where `NamespaceSwappingField` did not work correctly with `.empty_field()`.

### Added

- A method to ModelTestCase for running basic model tests when you aren't using config files.

### Added

- Added an option to `file_utils.cached_path` to automatically extract archives.
- Added the ability to pass an archive file instead of a local directory to `Vocab.from_files`.
- Added the ability to pass an archive file instead of a glob to `ShardedDatasetReader`.

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
