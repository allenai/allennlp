"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate --help
    usage: run [command] evaluate [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive_file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation_data_file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --cuda_device CUDA_DEVICE
                            id of GPU to use (if any)
"""

from inspect import signature
from typing import Dict, Any
import argparse
import logging

from allennlp.data import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables

import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Evaluate the specified model + dataset'''
    subparser = parser.add_parser(
            'evaluate', description=description, help='Evaluate the specified model + dataset')
    subparser.add_argument('--archive_file',
                           type=str,
                           required=True,
                           help='path to an archived trained model')
    subparser.add_argument('--evaluation_data_file',
                           type=str,
                           required=True,
                           help='path to the file containing the evaluation data')
    subparser.add_argument('--cuda_device',
                           type=int,
                           default=-1,
                           help='id of GPU to use (if any)')

    subparser.set_defaults(func=evaluate_from_args)

    return subparser

def evaluate(model: Model,
             dataset: Dataset,
             iterator: DataIterator,
             cuda_device: int) -> Dict[str, Any]:
    model.eval()

    generator = iterator(dataset, num_epochs=1)
    logger.info("Iterating over dataset")
    generator_tqdm = tqdm.tqdm(generator, total=iterator.get_num_batches(dataset))
    for batch in generator_tqdm:
        tensor_batch = arrays_to_variables(batch, cuda_device, for_training=False)
        if 'metadata' in tensor_batch and 'metadata' not in signature(model.forward).parameters:
            del tensor_batch['metadata']
        model.forward(**tensor_batch)
        metrics = model.get_metrics()
        description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
        generator_tqdm.set_description(description)

    return model.get_metrics()

def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device)
    config = archive.config
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(model.vocab)

    iterator = DataIterator.from_params(config.pop("iterator"))

    metrics = evaluate(model, dataset, iterator, args.cuda_device)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
