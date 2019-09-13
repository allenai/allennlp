"""
The ``find-lr`` subcommand can be used to find a good learning rate for a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp find-lr --help
    usage: allennlp find-lr [-h] -s SERIALIZATION_DIR [-o OVERRIDES]
                            [--start-lr START_LR] [--end-lr END_LR]
                            [--num-batches NUM_BATCHES]
                            [--stopping-factor STOPPING_FACTOR] [--linear] [-f]
                            [--include-package INCLUDE_PACKAGE]
                            param_path

    Find a learning rate range where loss decreases quickly for the specified
    model and dataset.

    positional arguments:
      param_path            path to parameter file describing the model to be
                            trained

    optional arguments:
      -h, --help            show this help message and exit
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            The directory in which to save results.
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --start-lr START_LR   learning rate to start the search (default = 1e-05)
      --end-lr END_LR       learning rate up to which search is done (default =
                            10)
      --num-batches NUM_BATCHES
                            number of mini-batches to run learning rate finder
                            (default = 100)
      --stopping-factor STOPPING_FACTOR
                            stop the search when the current loss exceeds the best
                            loss recorded by multiple of stopping factor
      --linear              increase learning rate linearly instead of exponential
                            increase
      -f, --force           overwrite the output directory if it exists
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""

from typing import List, Tuple
import argparse
import re
import os
import math
import logging
import shutil

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment, lazy_groups_of
from allennlp.data import Vocabulary, DataIterator
from allennlp.models import Model
from allennlp.training import Trainer
from allennlp.training.util import datasets_from_params


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FindLearningRate(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Find a learning rate range where loss decreases quickly
                         for the specified model and dataset.'''
        subparser = parser.add_parser(name,
                                      description=description,
                                      help='Find a learning rate range.')

        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')
        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='The directory in which to save results.')
        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')
        subparser.add_argument('--start-lr',
                               type=float,
                               default=1e-5,
                               help='learning rate to start the search')
        subparser.add_argument('--end-lr',
                               type=float,
                               default=10,
                               help='learning rate up to which search is done')
        subparser.add_argument('--num-batches',
                               type=int,
                               default=100,
                               help='number of mini-batches to run learning rate finder')
        subparser.add_argument('--stopping-factor',
                               type=float,
                               default=None,
                               help='stop the search when the current loss exceeds the best loss recorded by '
                                    'multiple of stopping factor')
        subparser.add_argument('--linear',
                               action='store_true',
                               help='increase learning rate linearly instead of exponential increase')
        subparser.add_argument('-f', '--force',
                               action='store_true',
                               required=False,
                               help='overwrite the output directory if it exists')

        subparser.set_defaults(func=find_learning_rate_from_args)

        return subparser

def find_learning_rate_from_args(args: argparse.Namespace) -> None:
    """
    Start learning rate finder for given args
    """
    params = Params.from_file(args.param_path, args.overrides)
    find_learning_rate_model(params, args.serialization_dir,
                             start_lr=args.start_lr,
                             end_lr=args.end_lr,
                             num_batches=args.num_batches,
                             linear_steps=args.linear,
                             stopping_factor=args.stopping_factor,
                             force=args.force)

def find_learning_rate_model(params: Params, serialization_dir: str,
                             start_lr: float = 1e-5,
                             end_lr: float = 10,
                             num_batches: int = 100,
                             linear_steps: bool = False,
                             stopping_factor: float = None,
                             force: bool = False) -> None:
    """
    Runs learning rate search for given `num_batches` and saves the results in ``serialization_dir``

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results.
    start_lr: ``float``
        Learning rate to start the search.
    end_lr: ``float``
        Learning rate upto which search is done.
    num_batches: ``int``
        Number of mini-batches to run Learning rate finder.
    linear_steps: ``bool``
        Increase learning rate linearly if False exponentially.
    stopping_factor: ``float``
        Stop the search when the current loss exceeds the best loss recorded by
        multiple of stopping factor. If ``None`` search proceeds till the ``end_lr``
    force: ``bool``
        If True and the serialization directory already exists, everything in it will
        be removed prior to finding the learning rate.
    """
    if os.path.exists(serialization_dir) and force:
        shutil.rmtree(serialization_dir)

    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        raise ConfigurationError(f'Serialization directory {serialization_dir} already exists and is '
                                 f'not empty.')
    else:
        os.makedirs(serialization_dir, exist_ok=True)

    prepare_environment(params)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation))
    vocab = Vocabulary.from_params(
            params.pop("vocabulary", {}),
            (instance for key, dataset in all_datasets.items()
             for instance in dataset
             if key in datasets_for_vocab_creation)
    )

    model = Model.from_params(vocab=vocab, params=params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)

    train_data = all_datasets['train']

    trainer_params = params.pop("trainer")
    no_grad_regexes = trainer_params.pop("no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)


    trainer_choice = trainer_params.pop("type", "default")
    if trainer_choice != "default":
        raise ConfigurationError("currently find-learning-rate only works with the default Trainer")
    trainer = Trainer.from_params(model=model,
                                  serialization_dir=serialization_dir,
                                  iterator=iterator,
                                  train_data=train_data,
                                  validation_data=None,
                                  params=trainer_params,
                                  validation_iterator=None)

    logger.info(f'Starting learning rate search from {start_lr} to {end_lr} in {num_batches} iterations.')
    learning_rates, losses = search_learning_rate(trainer,
                                                  start_lr=start_lr,
                                                  end_lr=end_lr,
                                                  num_batches=num_batches,
                                                  linear_steps=linear_steps,
                                                  stopping_factor=stopping_factor)
    logger.info(f'Finished learning rate search.')
    losses = _smooth(losses, 0.98)

    _save_plot(learning_rates, losses, os.path.join(serialization_dir, 'lr-losses.png'))

def search_learning_rate(trainer: Trainer,
                         start_lr: float = 1e-5,
                         end_lr: float = 10,
                         num_batches: int = 100,
                         linear_steps: bool = False,
                         stopping_factor: float = None) -> Tuple[List[float], List[float]]:
    """
    Runs training loop on the model using :class:`~allennlp.training.trainer.Trainer`
    increasing learning rate from ``start_lr`` to ``end_lr`` recording the losses.
    Parameters
    ----------
    trainer: :class:`~allennlp.training.trainer.Trainer`
    start_lr: ``float``
        The learning rate to start the search.
    end_lr: ``float``
        The learning rate upto which search is done.
    num_batches: ``int``
        Number of batches to run the learning rate finder.
    linear_steps: ``bool``
        Increase learning rate linearly if False exponentially.
    stopping_factor: ``float``
        Stop the search when the current loss exceeds the best loss recorded by
        multiple of stopping factor. If ``None`` search proceeds till the ``end_lr``
    Returns
    -------
    (learning_rates, losses): ``Tuple[List[float], List[float]]``
        Returns list of learning rates and corresponding losses.
        Note: The losses are recorded before applying the corresponding learning rate
    """
    if num_batches <= 10:
        raise ConfigurationError('The number of iterations for learning rate finder should be greater than 10.')

    trainer.model.train()

    num_gpus = len(trainer._cuda_devices) # pylint: disable=protected-access

    raw_train_generator = trainer.iterator(trainer.train_data,
                                           shuffle=trainer.shuffle)
    train_generator = lazy_groups_of(raw_train_generator, num_gpus)
    train_generator_tqdm = Tqdm.tqdm(train_generator,
                                     total=num_batches)

    learning_rates = []
    losses = []
    best = 1e9
    if linear_steps:
        lr_update_factor = (end_lr - start_lr) / num_batches
    else:
        lr_update_factor = (end_lr / start_lr) ** (1.0 / num_batches)

    for i, batch_group in enumerate(train_generator_tqdm):

        if linear_steps:
            current_lr = start_lr + (lr_update_factor * i)
        else:
            current_lr = start_lr * (lr_update_factor ** i)

        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = current_lr

        trainer.optimizer.zero_grad()
        loss = trainer.batch_loss(batch_group, for_training=True)
        loss.backward()
        loss = loss.detach().cpu().item()

        if stopping_factor is not None and (math.isnan(loss) or loss > stopping_factor * best):
            logger.info(f'Loss ({loss}) exceeds stopping_factor * lowest recorded loss.')
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
    """ Exponential smoothing of values """
    avg_value = 0.
    smoothed = []
    for i, value in enumerate(values):
        avg_value = beta * avg_value + (1 - beta) * value
        smoothed.append(avg_value / (1 - beta ** (i + 1)))
    return smoothed


def _save_plot(learning_rates: List[float], losses: List[float], save_path: str):
    # pylint: disable=multiple-statements,wrong-import-position
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ylabel('loss')
    plt.xlabel('learning rate (log10 scale)')
    plt.xscale('log')
    plt.plot(learning_rates, losses)
    logger.info(f'Saving learning_rate vs loss plot to {save_path}.')
    plt.savefig(save_path)
