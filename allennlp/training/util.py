"""
Helper functions for Trainers
"""
from typing import Any, Union, Dict, Iterable, List, Optional
import datetime
import logging
import os
import shutil

import torch
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import gather

from allennlp.common.checks import ConfigurationError, check_for_data_path, check_for_gpu
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import scatter_kwargs
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.models.model import Model
from allennlp.models.archival import CONFIG_NAME
from allennlp.nn import util as nn_util

logger = logging.getLogger(__name__)

# We want to warn people that tqdm ignores metrics that start with underscores
# exactly once. This variable keeps track of whether we have.
class HasBeenWarned:
    tqdm_ignores_underscores = False

def sparse_clip_norm(parameters, max_norm, norm_type=2) -> float:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Supports sparse gradients.

    Parameters
    ----------
    parameters : ``(Iterable[torch.Tensor])``
        An iterable of Tensors that will have gradients normalized.
    max_norm : ``float``
        The max norm of the gradients.
    norm_type : ``float``
        The type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns
    -------
    Total norm of the parameters (viewed as a single vector).
    """
    # pylint: disable=invalid-name,protected-access
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            if p.grad.is_sparse:
                # need to coalesce the repeated indices before finding norm
                grad = p.grad.data.coalesce()
                param_norm = grad._values().norm(norm_type)
            else:
                param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad.is_sparse:
                p.grad.data._values().mul_(clip_coef)
            else:
                p.grad.data.mul_(clip_coef)
    return total_norm


def move_optimizer_to_cuda(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if isinstance(param_state[k], torch.Tensor):
                        param_state[k] = param_state[k].cuda(device=param.get_device())


def get_batch_size(batch: Union[Dict, torch.Tensor]) -> int:
    """
    Returns the size of the batch dimension. Assumes a well-formed batch,
    returns 0 otherwise.
    """
    if isinstance(batch, torch.Tensor):
        return batch.size(0) # type: ignore
    elif isinstance(batch, Dict):
        return get_batch_size(next(iter(batch.values())))
    else:
        return 0


def time_to_str(timestamp: int) -> str:
    """
    Convert seconds past Epoch to human readable string.
    """
    datetimestamp = datetime.datetime.fromtimestamp(timestamp)
    return '{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}'.format(
            datetimestamp.year, datetimestamp.month, datetimestamp.day,
            datetimestamp.hour, datetimestamp.minute, datetimestamp.second
    )


def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split('-')]
    return datetime.datetime(*pieces)

def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.
    """
    for data_name in ["train_data_path", "validation_data_path", "test_data_path"]:
        data_path = params.get(data_name, None)
        if data_path is not None:
            check_for_data_path(data_path, data_name)

    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets

def create_serialization_dir(
        params: Params,
        serialization_dir: str,
        recover: bool,
        force: bool) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.

    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    force: ``bool``
        If ``True``, we will overwrite the serialization directory if it already exists.
    """
    if recover and force:
        raise ConfigurationError("Illegal arguments: both force and recover are true.")

    if os.path.exists(serialization_dir) and force:
        shutil.rmtree(serialization_dir)

    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            # Check whether any of the training configuration differs from the configuration we are
            # resuming.  If so, warn the user that training may fail.
            fail = False
            flat_params = params.as_flat_dict()
            flat_loaded = loaded_params.as_flat_dict()
            for key in flat_params.keys() - flat_loaded.keys():
                logger.error(f"Key '{key}' found in training configuration but not in the serialization "
                             f"directory we're recovering from.")
                fail = True
            for key in flat_loaded.keys() - flat_params.keys():
                logger.error(f"Key '{key}' found in the serialization directory we're recovering from "
                             f"but not in the training config.")
                fail = True
            for key in flat_params.keys():
                if flat_params.get(key, None) != flat_loaded.get(key, None):
                    logger.error(f"Value for '{key}' in training configuration does not match that the value in "
                                 f"the serialization directory we're recovering from: "
                                 f"{flat_params[key]} != {flat_loaded[key]}")
                    fail = True
            if fail:
                raise ConfigurationError("Training configuration does not match the configuration we're "
                                         "recovering from.")
    else:
        if recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)

def data_parallel(batch, model: Model, cuda_devices: List) -> Dict[str, torch.Tensor]:
    """
    Performs a forward pass using multiple GPUs.  This is a simplification
    of torch.nn.parallel.data_parallel to support the allennlp model
    interface.
    """
    inputs, module_kwargs = scatter_kwargs((), batch, cuda_devices, 0)

    used_device_ids = cuda_devices[:len(inputs)]
    replicas = replicate(model, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)

    # Only the 'loss' is needed.
    # a (num_gpu, ) tensor with loss on each GPU
    losses = gather([output['loss'].unsqueeze(0) for output in outputs], used_device_ids[0], 0)
    return {'loss': losses.mean()}


def enable_gradient_clipping(model: Model, grad_clipping: Optional[float]) -> None:
    if grad_clipping is not None:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: nn_util.clamp_tensor(grad,
                                                                          minimum=-grad_clipping,
                                                                          maximum=grad_clipping))

def rescale_gradients(model: Model, grad_norm: Optional[float] = None) -> Optional[float]:
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    """
    if grad_norm:
        parameters_to_clip = [p for p in model.parameters()
                              if p.grad is not None]
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None

def get_metrics(model: Model, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
    """
    Gets the metrics but sets ``"loss"`` to
    the total loss divided by the ``num_batches`` so that
    the ``"loss"`` metric is "average loss per batch".
    """
    metrics = model.get_metrics(reset=reset)
    metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
    return metrics


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             batch_weight_key: str) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        # Number of batches in instances.
        batch_count = 0
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative weighted loss
        total_loss = 0.0
        # Cumulative weight across all batches.
        total_weight = 0.0

        for batch in generator_tqdm:
            batch_count += 1
            batch = nn_util.move_to_device(batch, cuda_device)
            output_dict = model(**batch)
            loss = output_dict.get("loss")

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                if batch_weight_key:
                    weight = output_dict[batch_weight_key].item()
                else:
                    weight = 1.0

                total_weight += weight
                total_loss += loss.item() * weight
                # Report the average loss so far.
                metrics["loss"] = total_loss / total_weight

            if (not HasBeenWarned.tqdm_ignores_underscores and
                        any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                HasBeenWarned.tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError("The model you are trying to evaluate only sometimes " +
                                   "produced a loss!")
            final_metrics["loss"] = total_loss / total_weight

        return final_metrics

def description_from_metrics(metrics: Dict[str, float]) -> str:
    if (not HasBeenWarned.tqdm_ignores_underscores and
                any(metric_name.startswith("_") for metric_name in metrics)):
        logger.warning("Metrics with names beginning with \"_\" will "
                       "not be logged to the tqdm progress bar.")
        HasBeenWarned.tqdm_ignores_underscores = True
    return ', '.join(["%s: %.4f" % (name, value)
                      for name, value in
                      metrics.items() if not name.startswith("_")]) + " ||"
