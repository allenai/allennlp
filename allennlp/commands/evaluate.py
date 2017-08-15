from typing import Dict, Any
import argparse
import json
import os
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params, replace_none
from allennlp.data import Vocabulary, Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables, device_mapping

import torch
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
    # Load parameter file
    with open(args.config_file) as config_file:
        config = Params(replace_none(json.loads(config_file.read())))

    # Find out where the model was serialized to
    serialization_prefix = config.get('trainer', {}).get('serialization_prefix')
    if serialization_prefix is None:
        raise ConfigurationError("trainer.serialization_prefix must be specified in config")


    # Load vocabulary from file
    vocab_dir = os.path.join(serialization_prefix, 'vocabulary')
    vocab = Vocabulary.from_files(vocab_dir)

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(vocab)

    # Set up GPU or CPU
    cuda_device = args.cuda_device

    # Instantiate model
    weights_file = args.weights_file or os.path.join(serialization_prefix, "best.th")

    model = Model.from_params(vocab, config.pop('model'))
    model_state = torch.load(weights_file, map_location=device_mapping(cuda_device))
    model.load_state_dict(model_state)
    model.eval()

    iterator = DataIterator.from_params(config.pop("iterator"))

    metrics = evaluate(model, dataset, iterator, cuda_device)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
