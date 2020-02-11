"""
Helper functions for Trainers
"""
import datetime
import logging
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.distributed as dist

from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import CONFIG_NAME
from allennlp.models.model import Model
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

    # Parameters

    parameters : `(Iterable[torch.Tensor])`
        An iterable of Tensors that will have gradients normalized.
    max_norm : `float`
        The max norm of the gradients.
    norm_type : `float`
        The type of the used p-norm. Can be `'inf'` for infinity norm.

    # Returns

    Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float("inf"):
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
        total_norm = total_norm ** (1.0 / norm_type)
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
        for param in param_group["params"]:
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
        return batch.size(0)  # type: ignore
    elif isinstance(batch, Dict):
        return get_batch_size(next(iter(batch.values())))
    else:
        return 0


def time_to_str(timestamp: int) -> str:
    """
    Convert seconds past Epoch to human readable string.
    """
    datetimestamp = datetime.datetime.fromtimestamp(timestamp)
    return "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
        datetimestamp.year,
        datetimestamp.month,
        datetimestamp.day,
        datetimestamp.hour,
        datetimestamp.minute,
        datetimestamp.second,
    )


def str_to_time(time_str: str) -> datetime.datetime:
    """
    Convert human readable string to datetime.datetime.
    """
    pieces: Any = [int(piece) for piece in time_str.split("-")]
    return datetime.datetime(*pieces)


def read_all_datasets(
    train_data_path: str,
    dataset_reader: DatasetReader,
    validation_dataset_reader: DatasetReader = None,
    validation_data_path: str = None,
    test_data_path: str = None,
) -> Dict[str, Iterable[Instance]]:
    """
    Reads all datasets (perhaps lazily, if the corresponding dataset readers are lazy) and returns a
    dictionary mapping dataset name ("train", "validation" or "test") to the iterable resulting from
    `reader.read(filename)`.
    """

    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_dataset_reader = validation_dataset_reader or dataset_reader

    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
    """
    Load all the datasets specified by the config.

    # Parameters

    params : `Params`
    cache_directory : `str`, optional
        If given, we will instruct the `DatasetReaders` that we construct to cache their
        instances in this location (or read their instances from caches in this location, if a
        suitable cache already exists).  This is essentially a `base` directory for the cache, as
        we will additionally add the `cache_prefix` to this directory, giving an actual cache
        location of `cache_directory + cache_prefix`.
    cache_prefix : `str`, optional
        This works in conjunction with the `cache_directory`.  The idea is that the
        `cache_directory` contains caches for all different parameter settings, while the
        `cache_prefix` captures a specific set of parameters that led to a particular cache file.
        That is, if you change the tokenization settings inside your `DatasetReader`, you don't
        want to read cached data that used the old settings.  In order to avoid this, we compute a
        hash of the parameters used to construct each `DatasetReader` and use that as a "prefix"
        to the cache files inside the base `cache_directory`.  So, a given `input_file` would
        be cached essentially as `cache_directory + cache_prefix + input_file`, where you specify
        a `cache_directory`, the `cache_prefix` is based on the dataset reader parameters, and
        the `input_file` is whatever path you provided to `DatasetReader.read()`.  In order to
        allow you to give recognizable names to these prefixes if you want them, you can manually
        specify the `cache_prefix`.  Note that in some rare cases this can be dangerous, as we'll
        use the `same` prefix for both train and validation dataset readers.
    """
    dataset_reader_params = params.pop("dataset_reader")
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(
            validation_dataset_reader_params
        )

    train_data_path = params.pop("train_data_path")
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop("validation_data_path", None)
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
    params: Params, serialization_dir: str, recover: bool, force: bool
) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.

    # Parameters

    params : `Params`
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : `str`
        The directory in which to save results and logs.
    recover : `bool`
        If `True`, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    force : `bool`
        If `True`, we will overwrite the serialization directory if it already exists.
    """
    if recover and force:
        raise ConfigurationError("Illegal arguments: both force and recover are true.")

    if os.path.exists(serialization_dir) and force:
        shutil.rmtree(serialization_dir)

    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(
                f"Serialization directory ({serialization_dir}) already exists and is "
                f"not empty. Specify --recover to recover from an existing output folder."
            )

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError(
                "The serialization directory already exists but doesn't "
                "contain a config.json. You probably gave the wrong directory."
            )
        loaded_params = Params.from_file(recovered_config_file)

        # Check whether any of the training configuration differs from the configuration we are
        # resuming.  If so, warn the user that training may fail.
        fail = False
        flat_params = params.as_flat_dict()
        flat_loaded = loaded_params.as_flat_dict()
        for key in flat_params.keys() - flat_loaded.keys():
            logger.error(
                f"Key '{key}' found in training configuration but not in the serialization "
                f"directory we're recovering from."
            )
            fail = True
        for key in flat_loaded.keys() - flat_params.keys():
            logger.error(
                f"Key '{key}' found in the serialization directory we're recovering from "
                f"but not in the training config."
            )
            fail = True
        for key in flat_params.keys():
            if flat_params.get(key) != flat_loaded.get(key):
                logger.error(
                    f"Value for '{key}' in training configuration does not match that the value in "
                    f"the serialization directory we're recovering from: "
                    f"{flat_params[key]} != {flat_loaded[key]}"
                )
                fail = True
        if fail:
            raise ConfigurationError(
                "Training configuration does not match the configuration we're recovering from."
            )
    else:
        if recover:
            raise ConfigurationError(
                f"--recover specified but serialization_dir ({serialization_dir}) "
                "does not exist.  There is nothing to recover from."
            )
        os.makedirs(serialization_dir, exist_ok=True)


def enable_gradient_clipping(model: Model, grad_clipping: Optional[float]) -> None:
    if grad_clipping is not None:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(
                    lambda grad: nn_util.clamp_tensor(
                        grad, minimum=-grad_clipping, maximum=grad_clipping
                    )
                )


def rescale_gradients(model: Model, grad_norm: Optional[float] = None) -> Optional[float]:
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    """
    if grad_norm:
        parameters_to_clip = [p for p in model.parameters() if p.grad is not None]
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None


def get_metrics(
    model: Model,
    total_loss: float,
    num_batches: int,
    reset: bool = False,
    world_size: int = 1,
    cuda_device: Union[int, List] = 0,
) -> Dict[str, float]:
    """
    Gets the metrics but sets `"loss"` to
    the total loss divided by the `num_batches` so that
    the `"loss"` metric is "average loss per batch".
    """
    metrics = model.get_metrics(reset=reset)
    metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0

    if world_size > 1:
        # In distributed mode, average out all metrics across GPUs
        aggregated_metrics = {}
        for metric_name, metric_val in metrics.items():
            if isinstance(cuda_device, list):
                metric_tensor = torch.tensor(metric_val).to(torch.device(cuda_device[0]))
            else:
                metric_tensor = torch.tensor(metric_val).to(torch.device(cuda_device))
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
            reduced_metric = metric_tensor.item() / world_size
            aggregated_metrics[metric_name] = reduced_metric
        return aggregated_metrics
    else:
        return metrics


def evaluate(
    model: Model,
    instances: Iterable[Instance],
    data_iterator: DataIterator,
    cuda_device: int,
    batch_weight_key: str,
) -> Dict[str, Any]:
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances, num_epochs=1, shuffle=False)
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

            if not HasBeenWarned.tqdm_ignores_underscores and any(
                metric_name.startswith("_") for metric_name in metrics
            ):
                logger.warning(
                    'Metrics with names beginning with "_" will '
                    "not be logged to the tqdm progress bar."
                )
                HasBeenWarned.tqdm_ignores_underscores = True
            description = (
                ", ".join(
                    [
                        "%s: %.2f" % (name, value)
                        for name, value in metrics.items()
                        if not name.startswith("_")
                    ]
                )
                + " ||"
            )
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError(
                    "The model you are trying to evaluate only sometimes " + "produced a loss!"
                )
            final_metrics["loss"] = total_loss / total_weight

        return final_metrics


def description_from_metrics(metrics: Dict[str, float]) -> str:
    if not HasBeenWarned.tqdm_ignores_underscores and any(
        metric_name.startswith("_") for metric_name in metrics
    ):
        logger.warning(
            'Metrics with names beginning with "_" will ' "not be logged to the tqdm progress bar."
        )
        HasBeenWarned.tqdm_ignores_underscores = True
    return (
        ", ".join(
            [
                "%s: %.4f" % (name, value)
                for name, value in metrics.items()
                if not name.startswith("_")
            ]
        )
        + " ||"
    )


def make_vocab_from_params(
    params: Params, serialization_dir: str, print_statistics: bool = False
) -> Vocabulary:
    vocab_params = params.pop("vocabulary", {})
    os.makedirs(serialization_dir, exist_ok=True)
    vocab_dir = os.path.join(serialization_dir, "vocabulary")

    if os.path.isdir(vocab_dir) and os.listdir(vocab_dir) is not None:
        raise ConfigurationError(
            "The 'vocabulary' directory in the provided serialization directory is non-empty"
        )

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info(
        "From dataset instances, %s will be considered for vocabulary creation.",
        ", ".join(datasets_for_vocab_creation),
    )

    instances: Iterable[Instance] = (
        instance
        for key, dataset in all_datasets.items()
        if key in datasets_for_vocab_creation
        for instance in dataset
    )

    if print_statistics:
        instances = list(instances)

    vocab = Vocabulary.from_params(vocab_params, instances=instances)

    logger.info(f"writing the vocabulary to {vocab_dir}.")
    vocab.save_to_files(vocab_dir)
    logger.info("done creating vocab")

    if print_statistics:
        dataset = Batch(instances)
        dataset.index_instances(vocab)
        dataset.print_statistics()
        vocab.print_statistics()

    return vocab
