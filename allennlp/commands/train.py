import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy
from typing import Any, Dict, Union

import numpy
import pyhocon
import torch

from allennlp.common.checks import log_pytorch_version_info, ensure_pythonhashseed_set
from allennlp.common.params import Params, replace_none
from allennlp.common.tee_logger import TeeLogger
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Train the specified model on the specified dataset.'''
    subparser = parser.add_parser(
            'train', description=description, help='Train a model')
    subparser.add_argument('param_path',
                           type=str,
                           help='path to parameter file describing the model to be trained')
    subparser.set_defaults(func=_train_model_from_args)

    return subparser

def prepare_environment(params: Union[Params, Dict[str, Any]]):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.
    Parameters
    ----------
    params: Params object or dict, required.
        A ``Params`` object or dict holding the json parameters.
    """
    seed = params.pop("random_seed", 13370)
    numpy_seed = params.pop("numpy_seed", 1337)
    torch_seed = params.pop("pytorch_seed", 133)

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

    log_pytorch_version_info()


def _train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namepsace`` object to a string path.
    """
    train_model_from_file(args.param_path)


def train_model_from_file(parameter_filename: str):
    """
    A wrapper around :func:`train_model` which loads json from a file.
    Parameters
    ----------
    param_path: str, required.
        A json parameter file specifying an AllenNLP experiment.
    """
    # We need the python hashseed to be set if we're training a model
    ensure_pythonhashseed_set()

    # Set logging format

    param_dict = pyhocon.ConfigFactory.parse_file(parameter_filename)
    train_model(param_dict)


def train_model(param_dict: Dict[str, Any]):
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    param_dict: Dict[str, any], required.
        A parameter file specifying an AllenNLP Experiment.
    """
    params = Params(replace_none(param_dict))
    prepare_environment(params)

    log_dir = params.get("serialization_prefix", None)  # pylint: disable=no-member
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        sys.stdout = TeeLogger(os.path.join(log_dir, "_stdout.log"), sys.stdout)  # type: ignore
        sys.stderr = TeeLogger(os.path.join(log_dir, "_stderr.log"), sys.stderr)  # type: ignore
        handler = logging.FileHandler(os.path.join(log_dir, "_python_logging.log"))
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logging.getLogger().addHandler(handler)
        serialisation_params = deepcopy(params).as_dict(quiet=True)
        with open(os.path.join(log_dir, "_model_params.json"), "w") as param_file:
            json.dump(serialisation_params, param_file)

    # Now we begin assembling the required parts for the Trainer.
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    # TODO(Mark): work out how this is going to be built with different options.
    vocab = Vocabulary.from_dataset(train_data)
    if log_dir:
        vocab.save_to_files(os.path.join(log_dir, "vocabulary"))

    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

    train_data.index_instances(vocab)
    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        validation_data.index_instances(vocab)
    else:
        validation_data = None

    trainer = Trainer.from_params(model, optimizer, iterator,
                                  train_data, validation_data,
                                  params)
    trainer.train()
