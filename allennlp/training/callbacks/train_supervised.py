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
    """
    def __init__(self,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None) -> None:
        self.loss: torch.Tensor = 0.0
        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping

    @handle_event(Events.TRAINING_START)
    def enable_gradient_clipping(self, trainer: 'CallbackTrainer'):
        training_util.enable_gradient_clipping(trainer.model, self.grad_clipping)

    @handle_event(Events.BATCH_START)
    def zero_grad(self, trainer: 'CallbackTrainer'):
        # pylint: disable=no-self-use
        # TODO: gradient accumulation
        trainer.optimizer.zero_grad()

    @handle_event(Events.FORWARD)
    def compute_loss(self, trainer: 'CallbackTrainer'):
        self.loss = trainer.batch_loss(trainer.batch_group, for_training=True)

        if torch.isnan(self.loss):
            raise ValueError("nan loss encountered")

    @handle_event(Events.BACKWARD)
    def backpropagate_errors(self, trainer: 'CallbackTrainer'):
        self.loss.backward()
        trainer.train_loss += self.loss.item()

    @handle_event(Events.BACKWARD, priority=1000)
    def optimize(self, trainer: 'CallbackTrainer'):
        trainer.batch_grad_norm = training_util.rescale_gradients(trainer.model, self.grad_norm)
        trainer.optimizer.step()

        # Update the description with the latest metrics
        trainer.train_metrics = training_util.get_metrics(trainer.model,
                                                          trainer.train_loss,
                                                          trainer.batches_this_epoch)
