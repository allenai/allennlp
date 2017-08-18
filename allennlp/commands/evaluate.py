from typing import Dict, Any
import argparse
import json
import logging

from allennlp.common.params import Params, replace_none
from allennlp.data import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables

import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Evaluate the specified model + dataset'''
    subparser = parser.add_parser(
            'evaluate', description=description, help='Evaluate the specified model + dataset')
    subparser.add_argument('--config_file',
                           type=str,
                           required=True,
                           help='path to the configuration file that trained the model')
    subparser.add_argument('--weights_file',
                           type=str,
                           help=('path to the saved model weights '
                                 '(defaults to best.th in the config-specified serialization directory'))
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
    for batch in tqdm.tqdm(generator):
        tensor_batch = arrays_to_variables(batch, cuda_device, for_training=False)
        model.forward(**tensor_batch)

    return model.get_metrics()

def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load parameter file
    with open(args.config_file) as config_file:
        config = Params(replace_none(json.loads(config_file.read())))

    model = Model.from_files(config, None, args.weights_file, args.cuda_device)
    model.eval()

    vocab = model._vocab  # pylint: disable=protected-access

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(vocab)

    iterator = DataIterator.from_params(config.pop("iterator"))

    metrics = evaluate(model, dataset, iterator, args.cuda_device)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
