# pylint: disable=no-self-use
from typing import Optional, TYPE_CHECKING
import logging

import torch

from allennlp.training import util as training_util
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import

logger = logging.getLogger(__name__)


@Callback.register("train_supervised")
class TrainSupervised(Callback):
    """
    This callback actually does the forward / backward part of the training.

    Parameters
    ----------
    grad_norm : float, optional (default = None)
        If provided, we rescale the gradients before the optimization step.
    grad_clipping : float, optional (default = None)
        If provided, we use this to clip gradients in our model.
    gradient_accumulation_period : int, optional (default = 1)
        Accumulate gradients over this many batches before calling optimizer.step().
    """
    def __init__(self,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 gradient_accumulation_period: int = 1) -> None:
        self.loss: torch.Tensor = 0.0
        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping
        self.gradient_accumulation_period = gradient_accumulation_period
        self.accumulated_batches = 0

    @handle_event(Events.TRAINING_START)
    def enable_gradient_clipping(self, trainer: 'CallbackTrainer'):
        training_util.enable_gradient_clipping(trainer.model, self.grad_clipping)

        # This is necessary to prevent a crash in the gradient accumulation case.
        trainer.train_metrics["loss"] = 0.0

    @handle_event(Events.EPOCH_START)
    def zero_grad(self, trainer: 'CallbackTrainer'):
        trainer.optimizer.zero_grad()
        trainer.batches_this_epoch = 0

    @handle_event(Events.BATCH_START)
    def increment_counters(self, trainer: 'CallbackTrainer'):
        self.accumulated_batches += 1
        if self.accumulated_batches >= self.gradient_accumulation_period:
            trainer.batches_this_epoch += 1
            trainer.batch_num_total += 1

    @handle_event(Events.FORWARD)
    def compute_loss(self, trainer: 'CallbackTrainer'):
        self.loss = (trainer.batch_loss(trainer.batch_group, for_training=True) /
                     self.gradient_accumulation_period)
        if torch.isnan(self.loss):
            raise ValueError("nan loss encountered")

    @handle_event(Events.BACKWARD)
    def backpropagate_errors(self, trainer: 'CallbackTrainer'):
        self.loss.backward()
        trainer.train_loss += self.loss.item()

    def _optimize(self, trainer: 'CallbackTrainer'):
        trainer.batch_grad_norm = training_util.rescale_gradients(trainer.model, self.grad_norm)
        trainer.optimizer.step()

        # Update the description with the latest metrics
        trainer.train_metrics.update(
                training_util.get_metrics(trainer.model, trainer.train_loss, trainer.batches_this_epoch)
        )

        # And zero out the gradients
        trainer.optimizer.zero_grad()
        self.accumulated_batches = 0

    @handle_event(Events.BACKWARD, priority=1000)
    def optimize_batch(self, trainer: 'CallbackTrainer'):
        if self.accumulated_batches >= self.gradient_accumulation_period:
            self._optimize(trainer)

    @handle_event(Events.VALIDATE, priority=-1000)
    def optimize_last_batch(self, trainer: 'CallbackTrainer'):
        """
        There will be leftover gradients if the number of batches is not
        divisible by `gradient_accumulation_period`. We need to take an
        optimizer step for these as well. We do this with very early priority on
        the VALIDATE event, which happens right after the last batch.
        """
        if self.accumulated_batches > 0:
            self._optimize(trainer)
            # Have to increment counters separately
            trainer.batch_num_total += 1
            trainer.batches_this_epoch += 1
