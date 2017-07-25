from typing import Any, Dict, Optional, Union
import random
import logging
import sys
import os
import json
from copy import deepcopy

import pyhocon
import numpy
import torch

from allennlp.common.params import Params, replace_none

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

    from allennlp.common.checks import log_pytorch_version_info
    log_pytorch_version_info()


def execute_driver_from_file(param_path: str,
                             driver_operation_override: Optional[str] = None):
    """
    A wrapper around :func:`execute_driver` which loads json from a file.
    Parameters
    ----------
    param_path: str, required.
        A json parameter file specifying an AllenNLP experiment.
    driver_operation_override: Optional[str], optional, (default = None).
        Frequently, you will want to run the same parameters using different drivers,
        such as for training and evaluation. In order to specify this from the command
        line, this parameter will override the "operation" key in the parameter JSON.
    """
    param_dict = pyhocon.ConfigFactory.parse_file(param_path)
    execute_driver(param_dict, driver_operation_override)


def execute_driver(param_dict: Dict[str, any],
                   driver_operation_override: Optional[str] = None):
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
    driver_operation_override: Optional[str], optional, (default = None).
        Frequently, you will want to run the same parameters using different drivers,
        such as for training and evaluation. In order to specify this from the command
        line, this parameter will override the "operation" key in the parameter JSON.
    """
    params = Params(replace_none(param_dict))
    prepare_environment(params)
    # These have to be imported _after_ we set the random seeds.
    # TODO(Mark): check this is correct/see if we need to move tensor.py out of common.
    from allennlp.common.tee_logger import TeeLogger
    from allennlp.experiments.driver import Driver
    from allennlp.experiments import Registry

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

    if driver_operation_override is None:
        driver = Driver.from_params(params)
    else:
        unused_driver_key = params.pop("operation", None)
        if unused_driver_key:
            logger.warning("driver_override: %s passed to execute_driver;"
                           " ignoring 'operation' key present in params: %s",
                           (driver_operation_override, unused_driver_key))
        driver = Registry.get_driver(driver_operation_override).from_params(params)
    driver.run()
