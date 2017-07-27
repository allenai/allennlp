from typing import Any, Dict, Union
import random
import logging
import sys
import os
import json
from copy import deepcopy

import pyhocon
import numpy
import torch

from allennlp.training.optimizers import get_optimizer_from_params
from allennlp.common.checks import log_pytorch_version_info
from allennlp.common.params import Params, replace_none
from allennlp.common.tee_logger import TeeLogger
from allennlp.data.dataset_reader import DatasetReader
from allennlp.data.data_iterator import DataIterator
from allennlp.training.trainer import Trainer
from allennlp.data import Vocabulary
from allennlp.training import Model


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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


def train_model_from_file(param_path: str):
    """
    A wrapper around :func:`execute_driver` which loads json from a file.
    Parameters
    ----------
    param_path: str, required.
        A json parameter file specifying an AllenNLP experiment.
    """
    param_dict = pyhocon.ConfigFactory.parse_file(param_path)
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
        sys.stdout = TeeLogger(log_dir + "_stdout.log", sys.stdout)
        sys.stderr = TeeLogger(log_dir + "_stderr.log", sys.stderr)
        handler = logging.FileHandler(log_dir + "_python_logging.log")
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
    train_data.index_instances(vocab)
    model = Model.from_params(vocab, params.pop('model'))

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        validation_data.index_instances(vocab)
    else:
        validation_data = None

    iterator = DataIterator.from_params(params.pop("iterator"))
    optimizer = get_optimizer_from_params(model.parameters(), params.pop("optimizer"))

    trainer = Trainer.from_params(model, optimizer, iterator,
                                  train_data, validation_data,
                                  params)

    trainer.train()
