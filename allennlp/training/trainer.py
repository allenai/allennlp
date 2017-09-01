"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""

import logging
import os
import shutil
import time
from inspect import signature
from typing import Dict, Optional, List  # pylint: disable=unused-import

import torch
import torch.optim.lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.optim.lr_scheduler import _LRScheduler as PytorchLRScheduler  # pylint: disable=protected-access
import tqdm
from tensorboard import SummaryWriter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Dataset
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables, device_mapping
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Trainer:
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Dataset,
                 validation_dataset: Optional[Dataset] = None,
                 patience: int = 2,
                 validation_metric: str = "-loss",
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 cuda_device: int = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[PytorchLRScheduler] = None,
                 no_tqdm: bool = False) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : int, optional (default=2)
            Number of epochs to be patient before early stopping.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device : int, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm : float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : PytorchLRScheduler, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.
        no_tqdm : ``bool``, optional (default=False)
            We use ``tqdm`` for logging, which will print a nice progress bar that updates in place
            after every batch.  This is nice if you're running training on a local shell, but can
            cause problems with log files from, e.g., a docker image running on kubernetes.  If
            ``no_tqdm`` is ``True``, we will not use tqdm, and instead log batch statistics using
            ``logger.info``, outputting a line at most every 10 seconds.
        """
        self._model = model
        self._iterator = iterator
        self._optimizer = optimizer
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset

        self._patience = patience
        self._num_epochs = num_epochs
        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler

        increase_or_decrease = validation_metric[0]
        if increase_or_decrease not in ["+", "-"]:
            raise ConfigurationError("Validation metrics must specify whether they should increase "
                                     "or decrease by pre-pending the metric name with a +/-.")
        self._validation_metric = validation_metric[1:]
        self._validation_metric_decreases = increase_or_decrease == "-"
        self._no_tqdm = no_tqdm

        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

        self._log_interval = 10  # seconds
        self._summary_interval = 100  # num batches between logging to tensorboard

    def train(self) -> None:
        epoch_counter = 0
        # Resume from serialization path if it contains a saved model.
        if self._serialization_dir is not None:
            # Set up tensorboard logging.
            train_log = SummaryWriter(os.path.join(self._serialization_dir, "log", "train"))
            validation_log = SummaryWriter(os.path.join(self._serialization_dir, "log", "validation"))
            if any(["model_state_epoch_" in x
                    for x in os.listdir(self._serialization_dir)]):
                logger.info("Loading model from checkpoint.")
                epoch_counter = self._restore_checkpoint()

        if self._grad_clipping is not None:
            # Pylint is unable to tell that we're in the case that _glad_clipping is not None...
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

        logger.info("Beginning training.")
        num_training_batches = self._iterator.get_num_batches(self._train_dataset)
        if self._validation_dataset is not None:
            num_validation_batches = self._iterator.get_num_batches(self._validation_dataset)
        validation_metric_per_epoch = []  # type: List[float]
        for epoch in range(epoch_counter, self._num_epochs):
            logger.info("Epoch %d/%d", epoch + 1, self._num_epochs)
            train_loss = 0.0
            val_loss = 0.0
            # Set the model to "train" mode.
            self._model.train()
            train_generator = self._iterator(self._train_dataset, num_epochs=1)

            train_generator_tqdm = tqdm.tqdm(train_generator,
                                             disable=self._no_tqdm,
                                             total=num_training_batches)
            last_log = time.time()
            batch_num = 0
            logger.info("Training")
            for batch in train_generator_tqdm:
                batch_num += 1
                self._optimizer.zero_grad()
                output_dict = self._forward(batch, for_training=True)
                try:
                    loss = output_dict["loss"]
                    loss.backward()
                    # Make sure Variable is on the cpu before converting to numpy.
                    # .cpu() is a no-op if you aren't using GPUs.
                    train_loss += loss.data.cpu().numpy()
                except KeyError:
                    raise ConfigurationError("The model you are trying to optimize does not contain a"
                                             " 'loss' key in the output of model.forward(inputs).")

                if self._grad_norm:
                    clip_grad_norm(self._model.parameters(), self._grad_norm)
                self._optimizer.step()
                metrics = self._model.get_metrics()
                metrics["loss"] = float(train_loss / batch_num)
                description = self._description_from_metrics(metrics)
                train_generator_tqdm.set_description(description)

                batch_num_total = num_training_batches * epoch + batch_num
                if self._serialization_dir and batch_num_total % self._summary_interval == 0:
                    for name, param in self._model.named_parameters():
                        train_log.add_scalar("PARAMETER_MEAN/" + name, param.data.mean(), batch_num_total)
                        train_log.add_scalar("PARAMETER_STD/" + name, param.data.std(), batch_num_total)
                        if param.grad is not None:
                            train_log.add_scalar("GRAD_MEAN/" + name, param.grad.data.mean(), batch_num_total)
                            train_log.add_scalar("GRAD_STD/" + name, param.grad.data.std(), batch_num_total)
                    train_log.add_scalar("LOSS/loss_train", metrics["loss"], batch_num_total)
                if self._no_tqdm and time.time() - last_log > self._log_interval:
                    logger.info("Batch %d/%d: %s", batch_num, num_training_batches, description)
                    last_log = time.time()
            metrics = self._model.get_metrics(reset=True)
            metrics["loss"] = float(train_loss / batch_num)

            if self._validation_dataset is not None:
                logger.info("Validating")
                # Switch to evaluation mode.
                self._model.eval()
                val_generator = self._iterator(self._validation_dataset, num_epochs=1)
                val_generator_tqdm = tqdm.tqdm(val_generator,
                                               disable=self._no_tqdm,
                                               total=num_validation_batches)
                batch_num = 0
                for batch in val_generator_tqdm:
                    batch_num += 1
                    val_output_dict = self._forward(batch, for_training=False)
                    loss = val_output_dict["loss"]
                    val_loss += loss.data.cpu().numpy()
                    val_metrics = self._model.get_metrics()
                    val_metrics["loss"] = float(val_loss / batch_num)
                    description = self._description_from_metrics(val_metrics)
                    val_generator_tqdm.set_description(description)
                    if self._no_tqdm and time.time() - last_log > self._log_interval:
                        logger.info("Batch %d/%d: %s", batch_num, num_validation_batches, description)
                        last_log = time.time()
                val_metrics = self._model.get_metrics(reset=True)
                val_metrics["loss"] = float(val_loss / batch_num)
                message_template = "Training %s : %3f    Validation %s : %3f "
                for name, value in metrics.items():
                    logger.info(message_template, name, value, name, val_metrics[name])
                    if self._serialization_dir:
                        train_log.add_scalar(name, value, epoch)
                        validation_log.add_scalar(name, val_metrics[name], epoch)

                this_epoch_val_metric = val_metrics[self._validation_metric]
                if len(validation_metric_per_epoch) > self._patience:
                    # Is the worst validation performance in past self._patience
                    # epochs is better than current value?
                    if self._validation_metric_decreases:
                        should_stop = max(validation_metric_per_epoch[-self._patience:]) < this_epoch_val_metric
                    else:
                        should_stop = min(validation_metric_per_epoch[-self._patience:]) > this_epoch_val_metric
                    if should_stop:
                        logger.info("Ran out of patience.  Stopping training.")
                        break
                validation_metric_per_epoch.append(this_epoch_val_metric)

                if self._validation_metric_decreases:
                    is_best_so_far = this_epoch_val_metric == min(validation_metric_per_epoch)
                else:
                    is_best_so_far = this_epoch_val_metric == max(validation_metric_per_epoch)
                if self._serialization_dir:
                    self._save_checkpoint(epoch, is_best=is_best_so_far)

                if self._learning_rate_scheduler:
                    # Grim hack to determine whether the validation metric we are recording
                    # needs to be passed to the scheduler. This is required because the
                    # step() function of the different schedulers are (understandably)
                    # different to ReduceLROnPlateau.
                    if isinstance(self._learning_rate_scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
                    self._learning_rate_scheduler.step(epoch)
            else:
                message_template = "Training %s : %3f "
                for name, value in metrics.items():
                    logger.info(message_template, name, value)
                    if self._serialization_dir:
                        train_log.add_scalar(name, value, epoch)
                if self._serialization_dir:
                    self._save_checkpoint(epoch)
                if self._learning_rate_scheduler:
                    if isinstance(self._learning_rate_scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        raise ConfigurationError("The reduce_on_plateau learning rate scheduler requires "
                                                 "a validation metric to compute the schedule and therefore "
                                                 "must be used with a validation dataset.")
                    self._learning_rate_scheduler.step(epoch)

    def _forward(self, batch: dict, for_training: bool) -> dict:
        tensor_batch = arrays_to_variables(batch, self._cuda_device, for_training=for_training)
        if 'metadata' in tensor_batch and 'metadata' not in signature(self._model.forward).parameters:
            del tensor_batch['metadata']
        return self._model.forward(**tensor_batch)

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        # pylint: disable=no-self-use
        return ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"

    def _save_checkpoint(self,
                         epoch: int,
                         is_best: Optional[bool] = None) -> None:
        """
        Parameters
        ----------
        epoch : int, required.
            The epoch of training.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch, 'optimizer': self._optimizer.state_dict()}
        torch.save(training_state, os.path.join(self._serialization_dir,
                                                "training_state_epoch_{}.th".format(epoch)))
        if is_best:
            logger.info("Best validation performance so far. "
                        "Copying weights to %s/best.th'.", self._serialization_dir)
            shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

    def _restore_checkpoint(self) -> int:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        Returns
        -------
        epoch: int
            The epoch at which to resume training.
        """
        if not self._serialization_dir:
            raise ConfigurationError("serialization_dir not specified - cannot "
                                     "restore a model without a directory path.")

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) for x in model_checkpoints])

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_epoch_{}.th".format(epoch_to_load))

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))
        training_state = torch.load(training_state_path)
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(training_state["optimizer"])
        return training_state["epoch"]

    @classmethod
    def from_params(cls,
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_dataset: Dataset,
                    validation_dataset: Optional[Dataset],
                    params: Params) -> 'Trainer':

        patience = params.pop("patience", 2)
        validation_metric = params.pop("validation_metric", "-loss")
        num_epochs = params.pop("num_epochs", 20)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        if cuda_device >= 0:
            model = model.cuda(cuda_device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None
        no_tqdm = params.pop("no_tqdm", False)

        params.assert_empty(cls.__name__)
        return Trainer(model, optimizer, iterator,
                       train_dataset, validation_dataset,
                       patience=patience,
                       validation_metric=validation_metric,
                       num_epochs=num_epochs,
                       serialization_dir=serialization_dir,
                       cuda_device=cuda_device,
                       grad_norm=grad_norm,
                       grad_clipping=grad_clipping,
                       learning_rate_scheduler=scheduler,
                       no_tqdm=no_tqdm)
