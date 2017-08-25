import logging
import os
import shutil
from typing import Dict, Optional, List  # pylint: disable=unused-import

import torch
from torch.nn.utils.clip_grad import clip_grad_norm
import tqdm
from tensorboard import SummaryWriter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Dataset
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables, device_mapping

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
                 serialization_prefix: Optional[str] = None,
                 cuda_device: int = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
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
        serialization_prefix : str, optional (default=None)
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
        no_tqdm : ``bool``, optional (default=False)
            We use ``tqdm`` for logging, which will print a nice progress bar that updates in place
            after every batch.  This is nice if you're running training on a local shell, but can
            cause problems with log files from, e.g., a docker image running on kubernetes.  If
            ``no_tqdm`` is ``True``, we will not use tqdm, and instead log batch statistics using
            ``logger.info``.
        """
        self._model = model
        self._iterator = iterator
        self._optimizer = optimizer
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset

        self._patience = patience
        self._num_epochs = num_epochs
        self._serialization_prefix = serialization_prefix
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._log_interval = 100

        increase_or_decrease = validation_metric[0]
        if increase_or_decrease not in ["+", "-"]:
            raise ConfigurationError("Validation metrics must specify whether they should increase "
                                     "or decrease by pre-pending the metric name with a +/-.")
        self._validation_metric = validation_metric[1:]
        self._validation_metric_decreases = increase_or_decrease == "-"
        self._no_tqdm = no_tqdm

        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

    def train(self) -> None:
        epoch_counter = 0
        # Resume from serialization path if it contains a saved model.
        if self._serialization_prefix is not None:
            # Set up tensorboard logging.
            train_log = SummaryWriter(os.path.join(self._serialization_prefix, "log", "train"))
            validation_log = SummaryWriter(os.path.join(self._serialization_prefix, "log", "validation"))
            if any(["model_state_epoch_" in x
                    for x in os.listdir(self._serialization_prefix)]):
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
            batch_num = 0
            logger.info("Training")
            for batch in train_generator_tqdm:
                batch_num += 1
                tensor_batch = arrays_to_variables(batch, self._cuda_device)
                self._optimizer.zero_grad()
                output_dict = self._model.forward(**tensor_batch)
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
                batch_num_tot = num_training_batches * epoch + batch_num
                if self._serialization_prefix and batch_num_tot % self._log_interval == 0:
                    for name, param in self._model.named_parameters():
                        train_log.add_scalar("PARAMETER_MEAN/" + name, param.data.mean(), batch_num_tot)
                        train_log.add_scalar("PARAMETER_STD/" + name, param.data.std(), batch_num_tot)
                        if param.grad is not None:
                            train_log.add_scalar("GRAD_MEAN/" + name, param.grad.data.mean(), batch_num_tot)
                            train_log.add_scalar("GRAD_STD/" + name, param.grad.data.std(), batch_num_tot)
                    train_log.add_scalar("LOSS/loss_train", metrics["loss"], batch_num_tot)
                if self._no_tqdm:
                    logger.info("Batch %d/%d: %s", batch_num, num_training_batches, description)
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
                    tensor_batch = arrays_to_variables(batch, self._cuda_device, for_training=False)
                    val_output_dict = self._model.forward(**tensor_batch)
                    loss = val_output_dict["loss"]
                    val_loss += loss.data.cpu().numpy()
                    val_metrics = self._model.get_metrics()
                    val_metrics["loss"] = float(val_loss / batch_num)
                    description = self._description_from_metrics(val_metrics)
                    val_generator_tqdm.set_description(description)
                    if self._no_tqdm:
                        logger.info("Batch %d/%d: %s", batch_num, num_validation_batches, description)
                val_metrics = self._model.get_metrics(reset=True)
                val_metrics["loss"] = float(val_loss / batch_num)
                message_template = "Training %s : %3f    Validation %s : %3f "
                for name, value in metrics.items():
                    logger.info(message_template, name, value, name, val_metrics[name])
                    if self._serialization_prefix:
                        train_log.add_scalar(name, value, epoch)
                        validation_log.add_scalar(name, val_metrics[name], epoch)

                this_epoch = val_metrics[self._validation_metric]
                if len(validation_metric_per_epoch) > self._patience:
                    # Is the worst validation performance in past self._patience
                    # epochs is better than current value?
                    if self._validation_metric_decreases:
                        should_stop = max(validation_metric_per_epoch[-self._patience:]) < this_epoch
                    else:
                        should_stop = min(validation_metric_per_epoch[-self._patience:]) > this_epoch
                    if should_stop:
                        logger.info("Ran out of patience.  Stopping training.")
                        break
                validation_metric_per_epoch.append(this_epoch)

                if self._validation_metric_decreases:
                    is_best_so_far = this_epoch == min(validation_metric_per_epoch)
                else:
                    is_best_so_far = this_epoch == max(validation_metric_per_epoch)
                if self._serialization_prefix:
                    self._save_checkpoint(epoch, is_best=is_best_so_far)
            else:
                message_template = "Training %s : %3f "
                for name, value in metrics.items():
                    logger.info(message_template, name, value)
                    if self._serialization_prefix:
                        train_log.add_scalar(name, value, epoch)
                if self._serialization_prefix:
                    self._save_checkpoint(epoch)

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
        model_path = os.path.join(self._serialization_prefix, "model_state_epoch_{}.th".format(epoch))
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch, 'optimizer': self._optimizer.state_dict()}
        torch.save(training_state, os.path.join(self._serialization_prefix,
                                                "training_state_epoch_{}.th".format(epoch)))
        if is_best:
            logger.info("Best validation performance so far. "
                        "Copying weights to %s/best.th'.", self._serialization_prefix)
            shutil.copyfile(model_path, os.path.join(self._serialization_prefix, "best.th"))

    def _restore_checkpoint(self) -> int:
        """
        Restores a model from a serialization_prefix to the last saved checkpoint.
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
        if not self._serialization_prefix:
            raise ConfigurationError("serialization_prefix not specified - cannot "
                                     "restore a model without a directory path.")

        serialization_files = os.listdir(self._serialization_prefix)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) for x in model_checkpoints])

        model_path = os.path.join(self._serialization_prefix,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._serialization_prefix,
                                           "training_state_epoch_{}.th".format(epoch_to_load))

        model_state = torch.load(model_path, map_location=device_mapping(self._cuda_device))
        training_state = torch.load(training_state_path)
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(training_state["optimizer"])
        return training_state["epoch"]

    @classmethod
    def from_params(cls,
                    model: Model,
                    optimizer: torch.optim.Optimizer,
                    iterator: DataIterator,
                    train_dataset: Dataset,
                    validation_dataset: Optional[Dataset],
                    params: Optional[Params] = None) -> 'Trainer':

        params = params or Params({})
        patience = params.pop("patience", 2)
        validation_metric = params.pop("validation_metric", "-loss")
        num_epochs = params.pop("num_epochs", 20)
        serialization_prefix = params.pop("serialization_prefix", None)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        no_tqdm = params.pop("no_tqdm", False)
        params.assert_empty(cls.__name__)
        return Trainer(model, optimizer, iterator,
                       train_dataset, validation_dataset,
                       patience=patience,
                       validation_metric=validation_metric,
                       num_epochs=num_epochs,
                       serialization_prefix=serialization_prefix,
                       cuda_device=cuda_device,
                       grad_norm=grad_norm,
                       grad_clipping=grad_clipping,
                       no_tqdm=no_tqdm)
