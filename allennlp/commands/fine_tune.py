"""
The ``fine-tune`` subcommand is used to continue training (or `fine-tune`) a model on a `different
dataset` than the one it was originally trained on.  It requires a saved model archive file, a path
to the data you will continue training with, and a directory in which to write the results.

.. code-block:: bash

   $ allennlp fine-tune --help
    usage: allennlp fine-tune [-h] -m MODEL_ARCHIVE -c CONFIG_FILE -s
                              SERIALIZATION_DIR [-o OVERRIDES] [--extend-vocab]
                              [--file-friendly-logging]
                              [--batch-weight-key BATCH_WEIGHT_KEY]
                              [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                              [--include-package INCLUDE_PACKAGE]

    Continues training a saved model on a new dataset.

    optional arguments:
      -h, --help            show this help message and exit
      -m MODEL_ARCHIVE, --model-archive MODEL_ARCHIVE
                            path to the saved model archive from training on the
                            original data
      -c CONFIG_FILE, --config-file CONFIG_FILE
                            configuration file to use for training. Format is the
                            same as for the "train" command, but the "model"
                            section is ignored.
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the fine-tuned model and
                            its logs
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the training
                            configuration (only affects the config_file, _not_ the
                            model_archive)
      --extend-vocab        if specified, we will use the instances in your new
                            dataset to extend your vocabulary. If pretrained-file
                            was used to initialize embedding layers, you may also
                            need to pass --embedding-sources-mapping.
      --file-friendly-logging
                            outputs tqdm status on separate lines and slows tqdm
                            refresh rate
      --batch-weight-key BATCH_WEIGHT_KEY
                            If non-empty, name of metric used to weight the loss
                            on a per-batch basis.
      --embedding-sources-mapping EMBEDDING_SOURCES_MAPPING
                            a JSON dict defining mapping from embedding module
                            path to embedding pretrained-file used during
                            training. If not passed, and embedding needs to be
                            extended, we will try to use the original file paths
                            used during training. If they are not available we
                            will use random vectors for embedding extension.
                            (default = {})
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
import argparse
import json
import logging
from typing import Dict

from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.models import load_archive
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


class FineTune(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = """Continues training a saved model on a new dataset."""
        subparser = parser.add_parser(
            name, description=description, help="Continue training a model on a new dataset."
        )

        subparser.add_argument(
            "-m",
            "--model-archive",
            required=True,
            type=str,
            help="path to the saved model archive from training on the original data",
        )

        subparser.add_argument(
            "-c",
            "--config-file",
            required=True,
            type=str,
            help="configuration file to use for training. Format is the same as "
            'for the "train" command, but the "model" section is ignored.',
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the fine-tuned model and its logs",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the training configuration "
            "(only affects the config_file, _not_ the model_archive)",
        )

        subparser.add_argument(
            "--extend-vocab",
            action="store_true",
            default=False,
            help="if specified, we will use the instances in your new dataset to "
            "extend your vocabulary. If pretrained-file was used to initialize "
            "embedding layers, you may also need to pass --embedding-sources-mapping.",
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "--batch-weight-key",
            type=str,
            default="",
            help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
        )

        subparser.add_argument(
            "--embedding-sources-mapping",
            type=json.loads,
            default="{}",
            help="a JSON dict defining mapping from embedding module path to embedding "
            "pretrained-file used during training. If not passed, and embedding needs to be "
            "extended, we will try to use the original file paths used during training. If "
            "they are not available we will use random vectors for embedding extension.",
        )
        subparser.set_defaults(func=fine_tune_model_from_args)

        return subparser


def fine_tune_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    fine_tune_model_from_file_paths(
        model_archive_path=args.model_archive,
        config_file=args.config_file,
        serialization_dir=args.serialization_dir,
        overrides=args.overrides,
        extend_vocab=args.extend_vocab,
        file_friendly_logging=args.file_friendly_logging,
        batch_weight_key=args.batch_weight_key,
        embedding_sources_mapping=args.embedding_sources_mapping,
    )


def fine_tune_model_from_file_paths(
    model_archive_path: str,
    config_file: str,
    serialization_dir: str,
    overrides: str = "",
    extend_vocab: bool = False,
    file_friendly_logging: bool = False,
    recover: bool = False,
    force: bool = False,
    batch_weight_key: str = "",
    embedding_sources_mapping: Dict[str, str] = None,
) -> Model:
    """
    A wrapper around :func:`fine_tune_model` which loads the model archive from a file.

    # Parameters

    model_archive_path : ``str``
        Path to a saved model archive that is the result of running the ``train`` command.
    config_file : ``str``
        A configuration file specifying how to continue training.  The format is identical to the
        configuration file for the ``train`` command, but any contents in the ``model`` section is
        ignored (as we are using the provided model archive instead).
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`fine_tune_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    extend_vocab : ``bool``, optional (default=False)
        If ``True``, we use the new instances to extend your vocabulary.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`fine_tune_model`.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    batch_weight_key : ``str``, optional (default="")
        If non-empty, name of metric used to weight the loss on a per-batch basis.
    embedding_sources_mapping : ``Dict[str, str]``, optional (default=None)
        Mapping from model paths to the pretrained embedding filepaths.
    """
    # We don't need to pass in `cuda_device` here, because the trainer will call `model.cuda()` if
    # necessary.
    archive = load_archive(model_archive_path)
    params = Params.from_file(config_file, overrides)
    return train_model(
        model=archive.model,
        params=params,
        serialization_dir=serialization_dir,
        extend_vocab=extend_vocab,
        file_friendly_logging=file_friendly_logging,
        recover=recover,
        force=force,
        batch_weight_key=batch_weight_key,
        embedding_sources_mapping=embedding_sources_mapping,
    )
