"""
The `find-lr` subcommand can be used to find a good learning rate for a model.
It requires a configuration file and a directory in
which to write the results.
"""

import argparse
import logging
import math
import os
import re
from typing import List, Tuple
import itertools

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params, Tqdm
from allennlp.common import logging as common_logging
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training import GradientDescentTrainer, Trainer
from allennlp.training.util import create_serialization_dir, data_loaders_from_params

logger = logging.getLogger(__name__)


@Subcommand.register("find-lr")
class FindLearningRate(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:

        description = """Find a learning rate range where loss decreases quickly
                         for the specified model and dataset."""
        subparser = parser.add_parser(
            self.name, description=description, help="Find a learning rate range."
        )

        subparser.add_argument(
            "param_path", type=str, help="path to parameter file describing the model to be trained"
        )
        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="The directory in which to save results.",
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
            "--start-lr", type=float, default=1e-5, help="learning rate to start the search"
        )
        subparser.add_argument(
            "--end-lr", type=float, default=10, help="learning rate up to which search is done"
        )
        subparser.add_argument(
            "--num-batches",
            type=int,
            default=100,
            help="number of mini-batches to run learning rate finder",
        )
        subparser.add_argument(
            "--stopping-factor",
            type=float,
            default=None,
            help="stop the search when the current loss exceeds the best loss recorded by "
            "multiple of stopping factor",
        )
        subparser.add_argument(
            "--linear",
            action="store_true",
            help="increase learning rate linearly instead of exponential increase",
        )
        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=find_learning_rate_from_args)

        return subparser


def find_learning_rate_from_args(args: argparse.Namespace) -> None:
    """
    Start learning rate finder for given args
    """
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging
    params = Params.from_file(args.param_path, args.overrides)
    find_learning_rate_model(
        params,
        args.serialization_dir,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_batches=args.num_batches,
        linear_steps=args.linear,
        stopping_factor=args.stopping_factor,
        force=args.force,
    )


def find_learning_rate_model(
    params: Params,
    serialization_dir: str,
    start_lr: float = 1e-5,
    end_lr: float = 10,
    num_batches: int = 100,
    linear_steps: bool = False,
    stopping_factor: float = None,
    force: bool = False,
) -> None:
    """
    Runs learning rate search for given `num_batches` and saves the results in ``serialization_dir``

    # Parameters

    params : `Params`
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : `str`
        The directory in which to save results.
    start_lr : `float`
        Learning rate to start the search.
    end_lr : `float`
        Learning rate upto which search is done.
    num_batches : `int`
        Number of mini-batches to run Learning rate finder.
    linear_steps : `bool`
        Increase learning rate linearly if False exponentially.
    stopping_factor : `float`
        Stop the search when the current loss exceeds the best loss recorded by
        multiple of stopping factor. If `None` search proceeds till the `end_lr`
    force : `bool`
        If True and the serialization directory already exists, everything in it will
        be removed prior to finding the learning rate.
    """
    create_serialization_dir(params, serialization_dir, recover=False, force=force)

    prepare_environment(params)

    cuda_device = params.params.get("trainer").get("cuda_device", -1)
    check_for_gpu(cuda_device)
    distributed_params = params.params.get("distributed")
    # See https://github.com/allenai/allennlp/issues/3658
    assert not distributed_params, "find-lr is not compatible with DistributedDataParallel."

    all_data_loaders = data_loaders_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_data_loaders))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_data_loaders:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info(
        "From dataset instances, %s will be considered for vocabulary creation.",
        ", ".join(datasets_for_vocab_creation),
    )
    vocab = Vocabulary.from_params(
        params.pop("vocabulary", {}),
        instances=(
            instance
            for key, data_loader in all_data_loaders.items()
            if key in datasets_for_vocab_creation
            for instance in data_loader.iter_instances()
        ),
    )

    model = Model.from_params(
        vocab=vocab, params=params.pop("model"), serialization_dir=serialization_dir
    )

    all_data_loaders["train"].index_with(vocab)

    trainer_params = params.pop("trainer")

    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    trainer_choice = trainer_params.pop("type", "gradient_descent")
    if trainer_choice != "gradient_descent":
        raise ConfigurationError(
            "currently find-learning-rate only works with the GradientDescentTrainer"
        )
    trainer: GradientDescentTrainer = Trainer.from_params(  # type: ignore
        model=model,
        serialization_dir=serialization_dir,
        data_loader=all_data_loaders["train"],
        params=trainer_params,
    )

    logger.info(
        f"Starting learning rate search from {start_lr} to {end_lr} in {num_batches} iterations."
    )
    learning_rates, losses = search_learning_rate(
        trainer,
        start_lr=start_lr,
        end_lr=end_lr,
        num_batches=num_batches,
        linear_steps=linear_steps,
        stopping_factor=stopping_factor,
    )
    logger.info("Finished learning rate search.")
    losses = _smooth(losses, 0.98)

    _save_plot(learning_rates, losses, os.path.join(serialization_dir, "lr-losses.png"))


def search_learning_rate(
    trainer: GradientDescentTrainer,
    start_lr: float = 1e-5,
    end_lr: float = 10,
    num_batches: int = 100,
    linear_steps: bool = False,
    stopping_factor: float = None,
) -> Tuple[List[float], List[float]]:
    """
    Runs training loop on the model using [`GradientDescentTrainer`](../training/trainer.md#gradientdescenttrainer)
    increasing learning rate from `start_lr` to `end_lr` recording the losses.

    # Parameters

    trainer: `GradientDescentTrainer`
    start_lr : `float`
        The learning rate to start the search.
    end_lr : `float`
        The learning rate upto which search is done.
    num_batches : `int`
        Number of batches to run the learning rate finder.
    linear_steps : `bool`
        Increase learning rate linearly if False exponentially.
    stopping_factor : `float`
        Stop the search when the current loss exceeds the best loss recorded by
        multiple of stopping factor. If `None` search proceeds till the `end_lr`

    # Returns

    (learning_rates, losses) : `Tuple[List[float], List[float]]`
        Returns list of learning rates and corresponding losses.
        Note: The losses are recorded before applying the corresponding learning rate
    """
    if num_batches <= 10:
        raise ConfigurationError(
            "The number of iterations for learning rate finder should be greater than 10."
        )

    trainer.model.train()

    infinite_generator = itertools.cycle(trainer.data_loader)
    train_generator_tqdm = Tqdm.tqdm(infinite_generator, total=num_batches)

    learning_rates = []
    losses = []
    best = 1e9
    if linear_steps:
        lr_update_factor = (end_lr - start_lr) / num_batches
    else:
        lr_update_factor = (end_lr / start_lr) ** (1.0 / num_batches)

    for i, batch in enumerate(train_generator_tqdm):

        if linear_steps:
            current_lr = start_lr + (lr_update_factor * i)
        else:
            current_lr = start_lr * (lr_update_factor ** i)

        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = current_lr
            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self.optimizer.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for p in param_group["params"]:
                p.grad = None

        loss = trainer.batch_outputs(batch, for_training=True)["loss"]
        loss.backward()
        loss = loss.detach().cpu().item()

        if stopping_factor is not None and (math.isnan(loss) or loss > stopping_factor * best):
            logger.info(f"Loss ({loss}) exceeds stopping_factor * lowest recorded loss.")
            break

        trainer.rescale_gradients()
        trainer.optimizer.step()

        learning_rates.append(current_lr)
        losses.append(loss)

        if loss < best and i > 10:
            best = loss

        if i == num_batches:
            break

    return learning_rates, losses


def _smooth(values: List[float], beta: float) -> List[float]:
    """Exponential smoothing of values"""
    avg_value = 0.0
    smoothed = []
    for i, value in enumerate(values):
        avg_value = beta * avg_value + (1 - beta) * value
        smoothed.append(avg_value / (1 - beta ** (i + 1)))
    return smoothed


def _save_plot(learning_rates: List[float], losses: List[float], save_path: str):

    try:
        import matplotlib

        matplotlib.use("Agg")  # noqa
        import matplotlib.pyplot as plt

    except ModuleNotFoundError as error:

        logger.warn(
            "To use allennlp find-learning-rate, please install matplotlib: pip install matplotlib>=2.2.3 ."
        )
        raise error

    plt.ylabel("loss")
    plt.xlabel("learning rate (log10 scale)")
    plt.xscale("log")
    plt.plot(learning_rates, losses)
    logger.info(f"Saving learning_rate vs loss plot to {save_path}.")
    plt.savefig(save_path)
