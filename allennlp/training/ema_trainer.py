import logging
import os
import shutil
import time
from typing import Dict, Optional, List, Tuple, Union, Iterable

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.util import gpu_memory_mb, parse_cuda_device, peak_memory_mb
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer, time_to_str
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ExponentialMovingAverage:
    """
    Maintain Exponential Moving Average for model parameters.
    """
    def __init__(self, model: Model, decay: float = 0.9999):
        self.decay = decay
        self._average_values = {}
        self._backup_values = {}
        self._model = model
        for name, param in model.named_parameters():
            self._average_values[name] = param.data.clone()
            self._backup_values[name] = param.data.clone()

    def apply(self, num_updates: int = None, named_parameters: Iterable = None) -> None:
        """
        Apply exponential moving average to `named_parameters` if specified,
        or we will apply this to all the trainable parameters of the model.

        The optional `num_updates` parameter allows one to tweak the decay rate
        dynamically. If passed, the actual decay rate used is:

            `min(decay, (1 + num_updates) / (10 + num_updates))`

        """
        if num_updates is not None:
            decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        else:
            decay = self.decay
        if named_parameters is None:
            named_parameters = self._model.named_parameters()
        for name, param in named_parameters:
            new_average_value = (1.0 - decay) * param.data + decay * self._average_values[name]
            self._average_values[name] = new_average_value.clone()

    def assign_average_value(self, named_parameters=None) -> None:
        """
        Assign the exponential moving average value to the parameters
        """
        if named_parameters is None:
            named_parameters = self._model.named_parameters()
        for name, param in named_parameters:
            self._backup_values[name] = param.data.clone()
            param.data = self._average_values[name]

    def restore(self, named_parameters=None) -> None:
        """
        Restore the original values of each parameter
        """
        if named_parameters is None:
            named_parameters = self._model.named_parameters()
        for name, param in named_parameters:
            param.data = self._backup_values[name].clone()


@Trainer.register("ema_trainer")
class EMATrainer(Trainer):
    """
    This class is a subclass of `allennlp.training.trainer.Trainer`, but it supports maintaining
    the moving averages of all weights by employing an exponential decay. During training, we employ
    a shadow variable for each parameter, which maintains the moving average. During evaluation, we
    backup the original parameters and assign the moving averages to corresponding parameters.

    Be careful that when saving the checkpoint, we will save the moving averages of parameters. This
    is necessary because we want the saved model to perform as well as the validated model if we load
    it later. But this may cause problems if you restart the training from checkpoint.
    """
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 exponential_moving_average_decay: float = 0.9999) -> None:
        super().__init__(model, optimizer, iterator, train_dataset, validation_dataset,
                         patience, validation_metric, validation_iterator, shuffle, num_epochs,
                         serialization_dir, num_serialized_models_to_keep,
                         keep_serialized_model_every_num_seconds,
                         model_save_interval, cuda_device, grad_norm, grad_clipping,
                         learning_rate_scheduler, summary_interval, histogram_interval,
                         should_log_parameter_statistics, should_log_learning_rate)
        if exponential_moving_average_decay is not None:
            self.exp_moving_average = ExponentialMovingAverage(model, exponential_moving_average_decay)
        else:
            self.exp_moving_average = None

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Exactly the same as Trainer._train_epoch except for
        the addition of a call to self.exp_moving_average.apply() after each training step.
        """
        # pylint: disable=logging-fstring-interpolation
        logger.info(f"Epoch {epoch}/{self._num_epochs - 1}")
        logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
        for gpu, memory in gpu_memory_mb().items():
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        # Get tqdm for the training batches
        train_generator = self.iterator(self.train_data,
                                        num_epochs=1,
                                        shuffle=self.shuffle)
        num_training_batches = self.iterator.get_num_batches(self.train_data)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        if self._histogram_interval is not None:
            histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        for batch in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self._log_histograms_this_batch = self._histogram_interval is not None and (
                    batch_num_total % self._histogram_interval == 0)

            self.optimizer.zero_grad()

            loss = self.batch_loss(batch, for_training=True)
            loss.backward()

            train_loss += loss.item()

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)

            if self._log_histograms_this_batch:
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters()}
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7),
                                                       batch_num_total)
            else:
                self.optimizer.step()

            if self.exp_moving_average is not None:
                self.exp_moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = self._get_metrics(train_loss, batches_this_epoch)
            description = self._description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if batch_num_total % self._summary_interval == 0:
                if self._should_log_parameter_statistics:
                    self._parameter_and_gradient_statistics_to_tensorboard(batch_num_total, batch_grad_norm)
                if self._should_log_learning_rate:
                    self._learning_rates_to_tensorboard(batch_num_total)
                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"], batch_num_total)
                self._metrics_to_tensorboard(batch_num_total,
                                             {"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._log_histograms_this_batch:
                self._histograms_to_tensorboard(batch_num_total, histogram_parameters)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, time_to_str(int(last_save_time))), [], is_best=False)
        return self._get_metrics(train_loss, batches_this_epoch, reset=True)

    @overrides
    def _validation_loss(self) -> Tuple[float, int]:
        """
        Exactly the same as Trainer._validation_loss except for:
        1. the addition of a call to self.exp_moving_average.assign_average_value() before the
           validation so that the model can use the moving averages of parameters to do testing.
        2. the addition of a call to self.exp_moving_average.restore() after the validation so
           that the model parameters are restored to the original ones for training.
        """
        logger.info("Validating")

        self.model.eval()
        if self.exp_moving_average is not None:
            self.exp_moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        val_generator = val_iterator(self._validation_data,
                                     num_epochs=1,
                                     shuffle=False)
        num_validation_batches = val_iterator.get_num_batches(self._validation_data)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch in val_generator_tqdm:

            loss = self.batch_loss(batch, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = self._get_metrics(val_loss, batches_this_epoch)
            description = self._description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        if self.exp_moving_average is not None:
            self.exp_moving_average.restore()

        return val_loss, batches_this_epoch

    @overrides
    def _save_checkpoint(self,
                         epoch: Union[int, str],
                         val_metric_per_epoch: List[float],
                         is_best: Optional[bool] = None) -> None:
        """
        Exactly the same as Trainer._save_checkpoint except that we will save the moving averages
        of the parameters instead of the original value.
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))

            if self.exp_moving_average is not None:
                self.exp_moving_average.assign_average_value()
                model_state = self.model.state_dict()
                torch.save(model_state, model_path)
                self.exp_moving_average.restore()
            else:
                model_state = self.model.state_dict()
                torch.save(model_state, model_path)

            training_state = {'epoch': epoch,
                              'val_metric_per_epoch': val_metric_per_epoch,
                              'optimizer': self.optimizer.state_dict(),
                              'batch_num_total': self._batch_num_total}
            if self._learning_rate_scheduler is not None:
                training_state["learning_rate_scheduler"] = \
                    self._learning_rate_scheduler.lr_scheduler.state_dict()
            training_path = os.path.join(self._serialization_dir,
                                         "training_state_epoch_{}.th".format(epoch))
            torch.save(training_state, training_path)
            if is_best:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))

            if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
                self._serialized_paths.append([time.time(), model_path, training_path])
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    # Check to see if we should keep this checkpoint, if it has been longer
                    # then self._keep_serialized_model_every_num_seconds since the last
                    # kept checkpoint.
                    remove_path = True
                    if self._keep_serialized_model_every_num_seconds is not None:
                        save_time = paths_to_remove[0]
                        time_since_checkpoint_kept = save_time - self._last_permanent_saved_checkpoint_time
                        if time_since_checkpoint_kept > self._keep_serialized_model_every_num_seconds:
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_permanent_saved_checkpoint_time = save_time
                    if remove_path:
                        for fname in paths_to_remove[1:]:
                            os.remove(fname)

    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None) -> 'EMATrainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = \
            params.pop_int("keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        exponential_moving_average_decay = params.pop_float("exponential_moving_average_decay", 0.9999)
        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=scheduler,
                   num_serialized_models_to_keep=num_serialized_models_to_keep,
                   keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   exponential_moving_average_decay=exponential_moving_average_decay)
