"""
The ``fine-tune`` subcommand is used to continue training (or `fine-tune`) a model on a `different
dataset` than the one it was originally trained on.  It requires a saved model archive file, a path
to the data you will continue training with, and a directory in which to write the results.

Run ``python -m allennlp.run fine-tune --help`` for detailed usage information.
"""
import argparse
import json
import logging
import os
import sys
from copy import deepcopy

from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import datasets_from_params
from allennlp.common.tee_logger import TeeLogger
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models import Archive, load_archive, archive_model
from allennlp.models.archival import CONFIG_NAME
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class FineTune(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = """Continues training a saved model on a new dataset."""
        subparser = parser.add_parser(name,
                                      description=description,
                                      help='Continue training a model on a new dataset')

        subparser.add_argument('-m', '--model-archive',
                               required=True,
                               type=str,
                               help='path to the saved model archive from training on the original data')

        subparser.add_argument('-d', '--data-path',
                               required=True,
                               type=str,
                               help='path to data to use for continuing training')

        subparser.add_argument('--validation-data-path',
                               type=str,
                               help='path to validation data to use for continuing training')

        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the fine-tuned model and its logs')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')

        subparser.add_argument('--file-friendly-logging',
                               action='store_true',
                               default=False,
                               help='outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.set_defaults(func=fine_tune_model_from_args)

        return subparser


def fine_tune_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    # Import any additional modules needed (to register custom classes)
    for package_name in args.include_package:
        import_submodules(package_name)
    fine_tune_model_from_file_paths(model_archive_path=args.model_archive,
                                    train_data_path=args.data_path,
                                    serialization_dir=args.serialization_dir,
                                    overrides=args.overrides,
                                    file_friendly_logging=args.file_friendly_logging)


def fine_tune_model_from_file_paths(model_archive_path: str,
                                    train_data_path: str,
                                    serialization_dir: str,
                                    validation_data_path: str = None,
                                    overrides: str = "",
                                    file_friendly_logging: bool = False) -> Model:
    """
    A wrapper around :func:`fine_tune_model` which loads the model archive from a file.

    Parameters
    ----------
    model_archive_path : ``str``
        Path to a saved model archive that is the result of running the ``train`` command.
    data_path : ``str``
        The training data to use for continuing training the saved model.  We just pass this along
        to :func:`fine_tune_model`.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`fine_tune_model`.
    validation_data_path : ``str``, optional
        If given, the validation data to use when continuing training the saved model.  We just
        pass this along to :func:`fine_tune_model`.
    overrides : ``str``
        A HOCON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`fine_tune_model`.
    """
    # We don't need to pass in `cuda_device` here, because the trainer will call `model.cuda()` if
    # necessary.
    archive = load_archive(model_archive_path, overrides=overrides)
    return fine_tune_model(archive=archive,
                           train_data_path=train_data_path,
                           serialization_dir=serialization_dir,
                           validation_data_path=validation_data_path,
                           file_friendly_logging=file_friendly_logging)


def fine_tune_model(archive: Archive,
                    train_data_path: str,
                    serialization_dir: str,
                    validation_data_path: str = None,
                    file_friendly_logging: bool = False) -> Model:
    """
    Fine tunes the model in the given archive, using the the same configuration as found in the
    archive, except with the new data provided.

    Parameters
    ----------
    archive : ``Archive``
        A saved model archive that is the result of running the ``train`` command.
    train_data_path : ``str``
        Path to the training data to use for fine-tuning.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    validation_data_path : ``str``, optional
        Path to the validation data to use while fine-tuning.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    """
    params = archive.config
    model = archive.model
    prepare_environment(params)

    os.makedirs(serialization_dir)
    Tqdm.set_slower_interval(file_friendly_logging)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), # type: ignore
                           sys.stdout,
                           file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), # type: ignore
                           sys.stderr,
                           file_friendly_logging)
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, CONFIG_NAME), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    vocab = archive.model.vocab
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)

    params['train_data_path'] = train_data_path
    params['validation_data_path'] = validation_data_path
    params['test_data_path'] = None
    all_datasets = datasets_from_params(params)

    train_data = all_datasets['train']
    validation_data = all_datasets.get('validation')

    trainer_params = params.pop("trainer")
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params)
    metrics = trainer.train()

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    metrics_json = json.dumps(metrics, indent=2)
    with open(os.path.join(serialization_dir, "metrics.json"), "w") as metrics_file:
        metrics_file.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    return model
