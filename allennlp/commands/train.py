"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp train --help
    usage: allennlp train [-h] -s SERIALIZATION_DIR [-r] [-f] [-o OVERRIDES]
                          [--file-friendly-logging]
                          [--cache-directory CACHE_DIRECTORY]
                          [--cache-prefix CACHE_PREFIX] [--node-rank NODE_RANK]
                          [--include-package INCLUDE_PACKAGE]
                          param_path

    Train the specified model on the specified dataset.

    positional arguments:
      param_path            path to parameter file describing the model to be
                            trained

    optional arguments:
      -h, --help            show this help message and exit
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the model and its logs
      -r, --recover         recover training from the state in serialization_dir
      -f, --force           overwrite the output directory if it exists
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --file-friendly-logging
                            outputs tqdm status on separate lines and slows tqdm
                            refresh rate
      --cache-directory CACHE_DIRECTORY
                            Location to store cache of data preprocessing
      --cache-prefix CACHE_PREFIX
                            Prefix to use for data caching, giving current
                            parameter settings a name in the cache, instead of
                            computing a hash
      --node-rank NODE_RANK
                            Rank of this node in the distributed setup (default =
                            0)
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""

import argparse
import logging
import os
from typing import Optional, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.util import (
    prepare_environment,
    prepare_global_logging,
    dump_metrics,
    import_submodules,
)
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.util import create_serialization_dir, evaluate, make_vocab_from_params

logger = logging.getLogger(__name__)


class Train(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(name, description=description, help="Train a model.")

        subparser.add_argument(
            "param_path", type=str, help="path to parameter file describing the model to be trained"
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "--cache-directory",
            type=str,
            default="",
            help="Location to store cache of data preprocessing",
        )

        subparser.add_argument(
            "--cache-prefix",
            type=str,
            default="",
            help="Prefix to use for data caching, giving current parameter "
            "settings a name in the cache, instead of computing a hash",
        )

        subparser.add_argument(
            "--node-rank", type=int, default=0, help="Rank of this node in the distributed setup"
        )

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(
        args.param_path,
        args.serialization_dir,
        args.overrides,
        args.file_friendly_logging,
        args.recover,
        args.force,
        args.cache_directory,
        args.cache_prefix,
        args.node_rank,
        args.include_package,
    )


def train_model_from_file(
    parameter_filename: str,
    serialization_dir: str,
    overrides: str = "",
    file_friendly_logging: bool = False,
    recover: bool = False,
    force: bool = False,
    cache_directory: str = None,
    cache_prefix: str = None,
    node_rank: int = 0,
    include_package: List[str] = None,
) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    parameter_filename : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    cache_directory : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    cache_prefix : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    node_rank : ``int``, optional
        Rank of the current node in distributed training
    include_package: ``str``, optional
        In distributed mode, extra packages mentioned will be imported in trainer workers.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(
        params,
        serialization_dir,
        file_friendly_logging,
        recover,
        force,
        cache_directory,
        cache_prefix,
        node_rank,
        include_package,
    )


def train_model(
    params: Params,
    serialization_dir: str,
    file_friendly_logging: bool = False,
    recover: bool = False,
    force: bool = False,
    cache_directory: str = None,
    cache_prefix: str = None,
    node_rank: int = 0,
    include_package: List[str] = None,
) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.
    cache_directory : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    cache_prefix : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    node_rank: ``int``, optional
        Rank of the current node in distributed training
    include_package: ``List[str]``, optional
        In distributed mode, extra packages mentioned will be imported in trainer workers.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    distributed_params = params.params.pop("distributed", None)
    # If distributed isn't in the config and the config contains strictly
    # one cuda device, we just run a single training process.
    if distributed_params is None:
        model = _train_worker(
            process_rank=0,
            params=params,
            serialization_dir=serialization_dir,
            file_friendly_logging=file_friendly_logging,
            recover=recover,
            cache_directory=cache_directory,
            cache_prefix=cache_prefix,
            include_package=include_package,
        )
        archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        return model

    # Otherwise, we are running multiple processes for training.
    else:
        # We are careful here so that we can raise a good error if someone
        # passed the wrong thing - cuda_devices are required.
        device_ids = distributed_params.pop("cuda_devices", None)
        multi_device = isinstance(device_ids, list) and len(device_ids) > 1

        if not multi_device:
            raise ConfigurationError(
                "Multiple cuda devices need to be configured to run distributed training."
            )
        check_for_gpu(device_ids)

        master_addr = distributed_params.pop("master_address", "127.0.0.1")
        master_port = distributed_params.pop("master_port", 29500)
        num_procs = len(device_ids)
        num_nodes = distributed_params.pop("num_nodes", 1)
        world_size = num_nodes * num_procs

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)

        logging.info(
            f"Switching to distributed training mode since multiple GPUs are configured"
            f"Master is at: {master_addr}:{master_port} | Rank of this node: {node_rank} | "
            f"Number of workers in this node: {num_procs} | Number of nodes: {num_nodes} | "
            f"World size: {world_size}"
        )

        # Creating `Vocabulary` objects from workers could be problematic since the data iterators
        # in each worker will yield only `rank` specific instances. Hence it is safe to construct
        # the vocabulary and write it to disk before initializing the distributed context. The workers
        # will load the vocabulary from the path specified.
        make_vocab_from_params(params.duplicate(), serialization_dir)
        params["vocabulary"] = {
            "directory_path": os.path.join(serialization_dir, "vocabulary"),
            "extend": False,  # vocab extension would have been done above
        }

        mp.spawn(
            _train_worker,
            args=(
                params.duplicate(),
                serialization_dir,
                file_friendly_logging,
                recover,
                cache_directory,
                cache_prefix,
                include_package,
                node_rank,
                master_addr,
                master_port,
                world_size,
                device_ids,
            ),
            nprocs=num_procs,
        )
        model = Model.load(params, serialization_dir)
        return model


def _train_worker(
    process_rank: int,
    params: Params,
    serialization_dir: str,
    file_friendly_logging: bool = False,
    recover: bool = False,
    cache_directory: str = None,
    cache_prefix: str = None,
    include_package: List[str] = None,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    world_size: int = 1,
    distributed_device_ids: List[str] = None,
) -> Optional[Model]:
    """
    Helper to train the configured model/experiment. In distributed mode, this is spawned as a
    worker process. In a single GPU experiment, this returns the ``Model`` object and in distributed
    training, nothing is returned.

    Parameters
    ----------
    process_rank : ``int``
        The process index that is initialized using the GPU device id.
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    cache_directory : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    cache_prefix : ``str``, optional
        For caching data pre-processing.  See :func:`allennlp.training.util.datasets_from_params`.
    include_package: ``List[str]``, optional
        In distributed mode, since this function would have been spawned as a separate process,
        the extra imports need to be done again. NOTE: This does not have any effect in single
        GPU training.
    node_rank: ``int``, optional
        Rank of the node
    world_size: ``int``, optional
        The number of processes involved in distributed training.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_global_logging(
        serialization_dir, file_friendly_logging, rank=process_rank, world_size=world_size
    )
    prepare_environment(params)

    distributed = world_size > 1

    # not using `allennlp.common.util.is_master` as the process group is yet to be initialized
    master = process_rank == 0

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    if distributed:
        # Since the worker is spawned and not forked, the extra imports
        # need to be done again.
        if include_package is not None:
            for package_name in include_package:
                import_submodules(package_name)

        num_procs_per_node = len(distributed_device_ids)
        # The Unique identifier of the worker process among all the processes in the
        # distributed training group is computed here. This is used while initializing
        # the process group using `init_process_group`
        global_rank = node_rank * num_procs_per_node + process_rank

        # In distributed training, the configured device is always going to be a list.
        # The corresponding gpu id for the particular worker is obtained by picking the id
        # from the device list with the rank as index
        gpu_id = distributed_device_ids[process_rank]  # type: ignore

        # Till now, "cuda_device" might not be set in the trainer params.
        # But a worker trainer needs to only know about its specific GPU id.
        params["trainer"]["cuda_device"] = gpu_id
        params["trainer"]["world_size"] = world_size
        params["trainer"]["distributed"] = True

        torch.cuda.set_device(gpu_id)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=global_rank,
        )
        logging.info(
            f"Process group of world size {world_size} initialized "
            f"for distributed training in worker {global_rank}"
        )

    trainer_type = params.get("trainer", {}).get("type", "default")

    if trainer_type == "default":
        # Special logic to instantiate backward-compatible trainer.
        pieces = TrainerPieces.from_params(
            params, serialization_dir, recover, cache_directory, cache_prefix
        )
        trainer = Trainer.from_params(
            model=pieces.model,
            serialization_dir=serialization_dir,
            iterator=pieces.iterator,
            train_data=pieces.train_dataset,
            validation_data=pieces.validation_dataset,
            params=pieces.params,
            validation_iterator=pieces.validation_iterator,
        )

        evaluation_iterator = pieces.validation_iterator or pieces.iterator
        evaluation_dataset = pieces.test_dataset

    else:
        if evaluate_on_test:
            raise ValueError(
                "--evaluate-on-test only works with the default Trainer. "
                "If you're using the CallbackTrainer you can use a callback "
                "to evaluate at Events.TRAINING_END; otherwise you'll have "
                "to run allennlp evaluate separately."
            )

        trainer = TrainerBase.from_params(
            params, serialization_dir, recover, cache_directory, cache_prefix
        )
        evaluation_dataset = None

    params.assert_empty("base train command")

    try:
        if distributed:  # let the setup get ready for all the workers
            dist.barrier()

        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if master and os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info(
                "Training interrupted by the user. Attempting to create "
                "a model archive using the current best epoch weights."
            )
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    if master:
        if evaluation_dataset and evaluate_on_test:
            logger.info("The model will be evaluated using the best epoch weights.")
            test_metrics = evaluate(
                trainer.model,
                evaluation_dataset,
                evaluation_iterator,
                cuda_device=trainer.cuda_device,
                # TODO(brendanr): Pass in an arg following Joel's trainer refactor.
                batch_weight_key="",
            )

            for key, value in test_metrics.items():
                metrics["test_" + key] = value
        elif evaluation_dataset:
            logger.info(
                "To evaluate on the test set after training, pass the "
                "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
            )
        dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    if not distributed:
        return trainer.model

    return None  # to make mypy happy
