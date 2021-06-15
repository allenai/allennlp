"""
The `train` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.
"""

import argparse
import logging
import os
from os import PathLike
from typing import Any, Dict, List, Optional, Union
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params, Registrable, Lazy
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.meta import Meta, META_NAME
from allennlp.common import logging as common_logging
from allennlp.common import util as common_util
from allennlp.common.plugins import import_plugins
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataLoader
from allennlp.models.archival import archive_model, CONFIG_NAME, verify_include_in_archive
from allennlp.models.model import _DEFAULT_WEIGHTS, Model
from allennlp.training.trainer import Trainer
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)


@Subcommand.register("train")
class Train(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Train the specified model on the specified dataset."""
        subparser = parser.add_parser(self.name, description=description, help="Train a model.")

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
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--node-rank", type=int, default=0, help="rank of this node in the distributed setup"
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "do not train a model, but create a vocabulary, show dataset statistics and "
                "other training information"
            ),
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an `argparse.Namespace` object to string paths.
    """
    train_model_from_file(
        parameter_filename=args.param_path,
        serialization_dir=args.serialization_dir,
        overrides=args.overrides,
        recover=args.recover,
        force=args.force,
        node_rank=args.node_rank,
        include_package=args.include_package,
        dry_run=args.dry_run,
        file_friendly_logging=args.file_friendly_logging,
    )


def train_model_from_file(
    parameter_filename: Union[str, PathLike],
    serialization_dir: Union[str, PathLike],
    overrides: Union[str, Dict[str, Any]] = "",
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: List[str] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
) -> Optional[Model]:
    """
    A wrapper around [`train_model`](#train_model) which loads the params from a file.

    # Parameters

    parameter_filename : `str`
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : `str`
        The directory in which to save results and logs. We just pass this along to
        [`train_model`](#train_model).
    overrides : `Union[str, Dict[str, Any]]`, optional (default = `""`)
        A JSON string or a dict that we will use to override values in the input parameter file.
    recover : `bool`, optional (default=`False`)
        If `True`, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see `Model.from_archive`.
    force : `bool`, optional (default=`False`)
        If `True`, we will overwrite the serialization directory if it already exists.
    node_rank : `int`, optional
        Rank of the current node in distributed training
    include_package : `str`, optional
        In distributed mode, extra packages mentioned will be imported in trainer workers.
    dry_run : `bool`, optional (default=`False`)
        Do not train a model, but create a vocabulary, show dataset statistics and other training
        information.
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.

    # Returns

    best_model : `Optional[Model]`
        The model with the best epoch weights or `None` if in dry run.
    """
    # Load the experiment config from a file and pass it to `train_model`.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(
        params=params,
        serialization_dir=serialization_dir,
        recover=recover,
        force=force,
        node_rank=node_rank,
        include_package=include_package,
        dry_run=dry_run,
        file_friendly_logging=file_friendly_logging,
    )


def train_model(
    params: Params,
    serialization_dir: Union[str, PathLike],
    recover: bool = False,
    force: bool = False,
    node_rank: int = 0,
    include_package: List[str] = None,
    dry_run: bool = False,
    file_friendly_logging: bool = False,
) -> Optional[Model]:
    """
    Trains the model specified in the given [`Params`](../common/params.md#params) object, using the data
    and training parameters also specified in that object, and saves the results in `serialization_dir`.

    # Parameters

    params : `Params`
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : `str`
        The directory in which to save results and logs.
    recover : `bool`, optional (default=`False`)
        If `True`, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see `Model.from_archive`.
    force : `bool`, optional (default=`False`)
        If `True`, we will overwrite the serialization directory if it already exists.
    node_rank : `int`, optional
        Rank of the current node in distributed training
    include_package : `List[str]`, optional
        In distributed mode, extra packages mentioned will be imported in trainer workers.
    dry_run : `bool`, optional (default=`False`)
        Do not train a model, but create a vocabulary, show dataset statistics and other training
        information.
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.

    # Returns

    best_model : `Optional[Model]`
        The model with the best epoch weights or `None` if in dry run.
    """
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    training_util.create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    meta = Meta.new()
    meta.to_file(os.path.join(serialization_dir, META_NAME))

    include_in_archive = params.pop("include_in_archive", None)
    verify_include_in_archive(include_in_archive)

    distributed_params = params.params.pop("distributed", None)
    # If distributed isn't in the config and the config contains strictly
    # one cuda device, we just run a single training process.
    if distributed_params is None:
        model = _train_worker(
            process_rank=0,
            params=params,
            serialization_dir=serialization_dir,
            include_package=include_package,
            dry_run=dry_run,
            file_friendly_logging=file_friendly_logging,
        )

        if not dry_run:
            archive_model(serialization_dir, include_in_archive=include_in_archive)
        return model

    # Otherwise, we are running multiple processes for training.
    else:
        common_logging.prepare_global_logging(
            serialization_dir,
            rank=0,
            world_size=1,
        )

        # We are careful here so that we can raise a good error if someone
        # passed the wrong thing - cuda_devices are required.
        device_ids = distributed_params.pop("cuda_devices", None)
        multi_device = isinstance(device_ids, list) and len(device_ids) > 1
        num_nodes = distributed_params.pop("num_nodes", 1)

        if not (multi_device or num_nodes > 1):
            raise ConfigurationError(
                "Multiple cuda devices/nodes need to be configured to run distributed training."
            )
        check_for_gpu(device_ids)

        primary_addr = distributed_params.pop("primary_address", "127.0.0.1")
        if primary_addr in ("127.0.0.1", "0.0.0.0", "localhost"):
            # If running locally, we can automatically find an open port if one is not specified.
            primary_port = (
                distributed_params.pop("primary_port", None) or common_util.find_open_port()
            )
        else:
            # Otherwise we require that the port be specified.
            primary_port = distributed_params.pop("primary_port")

        num_procs = len(device_ids)
        world_size = num_nodes * num_procs

        # Creating `Vocabulary` objects from workers could be problematic since
        # the data loaders in each worker will yield only `rank` specific
        # instances. Hence it is safe to construct the vocabulary and write it
        # to disk before initializing the distributed context. The workers will
        # load the vocabulary from the path specified.
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        if recover:
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = training_util.make_vocab_from_params(
                params.duplicate(), serialization_dir, print_statistics=dry_run
            )
        params["vocabulary"] = {
            "type": "from_files",
            "directory": vocab_dir,
            "padding_token": vocab._padding_token,
            "oov_token": vocab._oov_token,
        }

        logging.info(
            "Switching to distributed training mode since multiple GPUs are configured | "
            f"Primary is at: {primary_addr}:{primary_port} | Rank of this node: {node_rank} | "
            f"Number of workers in this node: {num_procs} | Number of nodes: {num_nodes} | "
            f"World size: {world_size}"
        )

        mp.spawn(
            _train_worker,
            args=(
                params.duplicate(),
                serialization_dir,
                include_package,
                dry_run,
                node_rank,
                primary_addr,
                primary_port,
                world_size,
                device_ids,
                file_friendly_logging,
                include_in_archive,
            ),
            nprocs=num_procs,
        )
        if dry_run:
            return None
        else:
            archive_model(serialization_dir, include_in_archive=include_in_archive)
            model = Model.load(params, serialization_dir)
            return model


def _train_worker(
    process_rank: int,
    params: Params,
    serialization_dir: Union[str, PathLike],
    include_package: List[str] = None,
    dry_run: bool = False,
    node_rank: int = 0,
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
    world_size: int = 1,
    distributed_device_ids: List[int] = None,
    file_friendly_logging: bool = False,
    include_in_archive: List[str] = None,
) -> Optional[Model]:
    """
    Helper to train the configured model/experiment. In distributed mode, this is spawned as a
    worker process. In a single GPU experiment, this returns the `Model` object and in distributed
    training, nothing is returned.

    # Parameters

    process_rank : `int`
        The process index that is initialized using the GPU device id.
    params : `Params`
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : `str`
        The directory in which to save results and logs.
    include_package : `List[str]`, optional
        In distributed mode, since this function would have been spawned as a separate process,
        the extra imports need to be done again. NOTE: This does not have any effect in single
        GPU training.
    dry_run : `bool`, optional (default=`False`)
        Do not train a model, but create a vocabulary, show dataset statistics and other training
        information.
    node_rank : `int`, optional
        Rank of the node.
    primary_addr : `str`, optional (default=`"127.0.0.1"`)
        Address of the primary node for distributed training.
    primary_port : `str`, optional (default=`"29500"`)
        Port of the primary node for distributed training.
    world_size : `int`, optional
        The number of processes involved in distributed training.
    distributed_device_ids: `List[str]`, optional
        IDs of the devices used involved in distributed training.
    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    include_in_archive : `List[str]`, optional
        Paths relative to `serialization_dir` that should be archived in addition to the default ones.

    # Returns

    best_model : `Optional[Model]`
        The model with the best epoch weights or `None` if in distributed training or in dry run.
    """
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    common_logging.prepare_global_logging(
        serialization_dir,
        rank=process_rank,
        world_size=world_size,
    )
    common_util.prepare_environment(params)

    distributed = world_size > 1

    primary = process_rank == 0

    include_package = include_package or []

    if distributed:
        assert distributed_device_ids is not None

        # Since the worker is spawned and not forked, the extra imports need to be done again.
        # Both the ones from the plugins and the ones from `include_package`.
        import_plugins()
        for package_name in include_package:
            common_util.import_module_and_submodules(package_name)

        num_procs_per_node = len(distributed_device_ids)
        # The Unique identifier of the worker process among all the processes in the
        # distributed training group is computed here. This is used while initializing
        # the process group using `init_process_group`
        global_rank = node_rank * num_procs_per_node + process_rank

        # Number of processes per node is useful to know if a process
        # is a primary in the local node(node in which it is running)
        os.environ["ALLENNLP_PROCS_PER_NODE"] = str(num_procs_per_node)

        # In distributed training, the configured device is always going to be a list.
        # The corresponding gpu id for the particular worker is obtained by picking the id
        # from the device list with the rank as index
        gpu_id = distributed_device_ids[process_rank]  # type: ignore

        # Till now, "cuda_device" might not be set in the trainer params.
        # But a worker trainer needs to only know about its specific GPU id.
        params["trainer"]["cuda_device"] = gpu_id
        params["trainer"]["world_size"] = world_size
        params["trainer"]["distributed"] = True

        if gpu_id >= 0:
            torch.cuda.set_device(int(gpu_id))
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{primary_addr}:{primary_port}",
                world_size=world_size,
                rank=global_rank,
            )
        else:
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{primary_addr}:{primary_port}",
                world_size=world_size,
                rank=global_rank,
            )
        logging.info(
            f"Process group of world size {world_size} initialized "
            f"for distributed training in worker {global_rank}"
        )

    train_loop = TrainModel.from_params(
        params=params,
        serialization_dir=serialization_dir,
        local_rank=process_rank,
    )

    if dry_run:
        return None

    try:
        if distributed:  # let the setup get ready for all the workers
            dist.barrier()

        metrics = train_loop.run()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if primary and os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            best_weights_path = train_loop.trainer.get_best_weights_path()
            if best_weights_path is None:
                logging.info(
                    "Training interrupted by the user, and no best model has been saved. "
                    "No model archive created."
                )
            else:
                logging.info(
                    "Training interrupted by the user. Attempting to create "
                    "a model archive using the current best epoch weights."
                )
                archive_model(
                    serialization_dir,
                    weights=best_weights_path,
                    include_in_archive=include_in_archive,
                )
        raise

    if primary:
        train_loop.finish(metrics)

    if not distributed:
        return train_loop.model

    return None


class TrainModel(Registrable):
    """
    This class exists so that we can easily read a configuration file with the `allennlp train`
    command.  The basic logic is that we call `train_loop =
    TrainModel.from_params(params_from_config_file)`, then `train_loop.run()`.  This class performs
    very little logic, pushing most of it to the `Trainer` that has a `train()` method.  The
    point here is to construct all of the dependencies for the `Trainer` in a way that we can do
    it using `from_params()`, while having all of those dependencies transparently documented and
    not hidden in calls to `params.pop()`.  If you are writing your own training loop, you almost
    certainly should not use this class, but you might look at the code for this class to see what
    we do, to make writing your training loop easier.

    In particular, if you are tempted to call the `__init__` method of this class, you are probably
    doing something unnecessary.  Literally all we do after `__init__` is call `trainer.train()`.  You
    can do that yourself, if you've constructed a `Trainer` already.  What this class gives you is a
    way to construct the `Trainer` by means of a config file.  The actual constructor that we use
    with `from_params` in this class is `from_partial_objects`.  See that method for a description
    of all of the allowed top-level keys in a configuration file used with `allennlp train`.
    """

    default_implementation = "default"
    """
    The default implementation is registered as 'default'.
    """

    def __init__(
        self,
        serialization_dir: str,
        model: Model,
        trainer: Trainer,
        evaluation_data_loader: DataLoader = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = "",
    ) -> None:
        self.serialization_dir = serialization_dir
        self.model = model
        self.trainer = trainer
        self.evaluation_data_loader = evaluation_data_loader
        self.evaluate_on_test = evaluate_on_test
        self.batch_weight_key = batch_weight_key

    def run(self) -> Dict[str, Any]:
        return self.trainer.train()

    def finish(self, metrics: Dict[str, Any]):
        if self.evaluation_data_loader is not None and self.evaluate_on_test:
            logger.info("The model will be evaluated using the best epoch weights.")
            test_metrics = training_util.evaluate(
                self.model,
                self.evaluation_data_loader,
                cuda_device=self.trainer.cuda_device,
                batch_weight_key=self.batch_weight_key,
            )

            for key, value in test_metrics.items():
                metrics["test_" + key] = value
        elif self.evaluation_data_loader is not None:
            logger.info(
                "To evaluate on the test set after training, pass the "
                "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
            )
        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"), metrics, log=True
        )

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        local_rank: int,
        dataset_reader: DatasetReader,
        train_data_path: Any,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
        trainer: Lazy[Trainer],
        vocabulary: Lazy[Vocabulary] = Lazy(Vocabulary),
        datasets_for_vocab_creation: List[str] = None,
        validation_dataset_reader: DatasetReader = None,
        validation_data_path: Any = None,
        validation_data_loader: Lazy[DataLoader] = None,
        test_data_path: Any = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = "",
    ) -> "TrainModel":
        """
        This method is intended for use with our `FromParams` logic, to construct a `TrainModel`
        object from a config file passed to the `allennlp train` command.  The arguments to this
        method are the allowed top-level keys in a configuration file (except for the first three,
        which are obtained separately).

        You *could* use this outside of our `FromParams` logic if you really want to, but there
        might be easier ways to accomplish your goal than instantiating `Lazy` objects.  If you are
        writing your own training loop, we recommend that you look at the implementation of this
        method for inspiration and possibly some utility functions you can call, but you very likely
        should not use this method directly.

        The `Lazy` type annotations here are a mechanism for building dependencies to an object
        sequentially - the `TrainModel` object needs data, a model, and a trainer, but the model
        needs to see the data before it's constructed (to create a vocabulary) and the trainer needs
        the data and the model before it's constructed.  Objects that have sequential dependencies
        like this are labeled as `Lazy` in their type annotations, and we pass the missing
        dependencies when we call their `construct()` method, which you can see in the code below.

        # Parameters

        serialization_dir: `str`
            The directory where logs and model archives will be saved.

            In a typical AllenNLP configuration file, this parameter does not get an entry as a
            top-level key, it gets passed in separately.

        local_rank: `int`
            The process index that is initialized using the GPU device id.

            In a typical AllenNLP configuration file, this parameter does not get an entry as a
            top-level key, it gets passed in separately.

        dataset_reader: `DatasetReader`
            The `DatasetReader` that will be used for training and (by default) for validation.

        train_data_path: `str`
            The file (or directory) that will be passed to `dataset_reader.read()` to construct the
            training data.

        model: `Lazy[Model]`
            The model that we will train.  This is lazy because it depends on the `Vocabulary`;
            after constructing the vocabulary we call `model.construct(vocab=vocabulary)`.

        data_loader: `Lazy[DataLoader]`
            The data_loader we use to batch instances from the dataset reader at training and (by
            default) validation time. This is lazy because it takes a dataset in it's constructor.

        trainer: `Lazy[Trainer]`
            The `Trainer` that actually implements the training loop.  This is a lazy object because
            it depends on the model that's going to be trained.

        vocabulary: `Lazy[Vocabulary]`, optional (default=`Lazy(Vocabulary)`)
            The `Vocabulary` that we will use to convert strings in the data to integer ids (and
            possibly set sizes of embedding matrices in the `Model`).  By default we construct the
            vocabulary from the instances that we read.

        datasets_for_vocab_creation: `List[str]`, optional (default=`None`)
            If you pass in more than one dataset but don't want to use all of them to construct a
            vocabulary, you can pass in this key to limit it.  Valid entries in the list are
            "train", "validation" and "test".

        validation_dataset_reader: `DatasetReader`, optional (default=`None`)
            If given, we will use this dataset reader for the validation data instead of
            `dataset_reader`.

        validation_data_path: `str`, optional (default=`None`)
            If given, we will use this data for computing validation metrics and early stopping.

        validation_data_loader: `Lazy[DataLoader]`, optional (default=`None`)
            If given, the data_loader we use to batch instances from the dataset reader at
            validation and test time. This is lazy because it takes a dataset in it's constructor.

        test_data_path: `str`, optional (default=`None`)
            If given, we will use this as test data.  This makes it available for vocab creation by
            default, but nothing else.

        evaluate_on_test: `bool`, optional (default=`False`)
            If given, we will evaluate the final model on this data at the end of training.  Note
            that we do not recommend using this for actual test data in every-day experimentation;
            you should only very rarely evaluate your model on actual test data.

        batch_weight_key: `str`, optional (default=`""`)
            The name of metric used to weight the loss on a per-batch basis.  This is only used
            during evaluation on final test data, if you've specified `evaluate_on_test=True`.
        """
        # Train data loader.
        data_loaders: Dict[str, DataLoader] = {
            "train": data_loader.construct(reader=dataset_reader, data_path=train_data_path)
        }

        # Validation data loader.
        if validation_data_path is not None:
            validation_dataset_reader = validation_dataset_reader or dataset_reader
            if validation_data_loader is not None:
                data_loaders["validation"] = validation_data_loader.construct(
                    reader=validation_dataset_reader, data_path=validation_data_path
                )
            else:
                data_loaders["validation"] = data_loader.construct(
                    reader=validation_dataset_reader, data_path=validation_data_path
                )
                if getattr(data_loaders["validation"], "batches_per_epoch", None) is not None:
                    warnings.warn(
                        "Using 'data_loader' params to construct validation data loader since "
                        "'validation_data_loader' params not specified, but you have "
                        "'data_loader.batches_per_epoch' set which may result in different "
                        "validation datasets for each epoch.",
                        UserWarning,
                    )

        # Test data loader.
        if test_data_path is not None:
            test_dataset_reader = validation_dataset_reader or dataset_reader
            if validation_data_loader is not None:
                data_loaders["test"] = validation_data_loader.construct(
                    reader=test_dataset_reader, data_path=test_data_path
                )
            else:
                data_loaders["test"] = data_loader.construct(
                    reader=test_dataset_reader, data_path=test_data_path
                )

        if datasets_for_vocab_creation:
            for key in datasets_for_vocab_creation:
                if key not in data_loaders:
                    raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {key}")

            logger.info(
                "From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation),
            )

        instance_generator = (
            instance
            for key, data_loader in data_loaders.items()
            if datasets_for_vocab_creation is None or key in datasets_for_vocab_creation
            for instance in data_loader.iter_instances()
        )

        vocabulary_ = vocabulary.construct(instances=instance_generator)

        model_ = model.construct(vocab=vocabulary_, serialization_dir=serialization_dir)

        # Initializing the model can have side effect of expanding the vocabulary.
        # Save the vocab only in the primary. In the degenerate non-distributed
        # case, we're trivially the primary. In the distributed case this is safe
        # to do without worrying about race conditions since saving and loading
        # the vocab involves acquiring a file lock.
        if local_rank == 0:
            vocabulary_path = os.path.join(serialization_dir, "vocabulary")
            vocabulary_.save_to_files(vocabulary_path)

        for data_loader_ in data_loaders.values():
            data_loader_.index_with(model_.vocab)

        trainer_ = trainer.construct(
            serialization_dir=serialization_dir,
            model=model_,
            data_loader=data_loaders["train"],
            validation_data_loader=data_loaders.get("validation"),
            local_rank=local_rank,
        )
        assert trainer_ is not None

        return cls(
            serialization_dir=serialization_dir,
            model=model_,
            trainer=trainer_,
            evaluation_data_loader=data_loaders.get("test"),
            evaluate_on_test=evaluate_on_test,
            batch_weight_key=batch_weight_key,
        )


TrainModel.register("default", constructor="from_partial_objects")(TrainModel)
