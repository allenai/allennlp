"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate --help
    usage: python -m allennlp.run [command] evaluate [-h] --evaluation-data-file
                                                    EVALUATION_DATA_FILE
                                                    [--cuda-device CUDA_DEVICE]
                                                    [-o OVERRIDES]
                                                    [--include-package INCLUDE_PACKAGE]
                                                    archive_file

    Evaluate the specified model + dataset

    positional arguments:
    archive_file          path to an archived trained model

    optional arguments:
    -h, --help            show this help message and exit
    --evaluation-data-file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a HOCON structure used to override the experiment
                            configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import Dict, Any, Iterable
import argparse
import logging

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment, import_submodules
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Evaluate(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
                name, description=description, help='Evaluate the specified model + dataset')

        subparser.add_argument('archive_file', type=str, help='path to an archived trained model')

        evaluation_data_file = subparser.add_mutually_exclusive_group(required=True)
        evaluation_data_file.add_argument('--evaluation-data-file',
                                          type=str,
                                          help='path to the file containing the evaluation data')
        evaluation_data_file.add_argument('--evaluation_data_file',
                                          type=str,
                                          help=argparse.SUPPRESS)

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')
        cuda_device.add_argument('--cuda_device', type=int, help=argparse.SUPPRESS)

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int) -> Dict[str, Any]:
    model.eval()

    iterator = data_iterator(instances, num_epochs=1, cuda_device=cuda_device, for_training=False)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
    for batch in generator_tqdm:
        model(**batch)
        metrics = model.get_metrics()
        description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
        generator_tqdm.set_description(description, refresh=False)

    return model.get_metrics()


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Import any additional modules needed (to register custom classes)
    for package_name in args.include_package:
        import_submodules(package_name)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    iterator = DataIterator.from_params(config.pop("iterator"))
    iterator.index_with(model.vocab)

    metrics = evaluate(model, instances, iterator, args.cuda_device)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
