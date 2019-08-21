from typing import Optional, TYPE_CHECKING
import logging

from allennlp.training import util as training_util
from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events

if TYPE_CHECKING:
    from allennlp.training.callback_trainer import CallbackTrainer  # pylint:disable=unused-import

logger = logging.getLogger(__name__)


@Callback.register("gradient_norm_and_clip")
class GradientNormAndClip(Callback):
    """
    Applies gradient norm and/or clipping.

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
        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping

    @handle_event(Events.TRAINING_START)
    def enable_gradient_clipping(self, trainer: 'CallbackTrainer'):
        training_util.enable_gradient_clipping(trainer.model, self.grad_clipping)

    @handle_event(Events.BACKWARD, priority=1000)
    def rescale_gradients(self, trainer: 'CallbackTrainer'):
        trainer.batch_grad_norm = training_util.rescale_gradients(trainer.model, self.grad_norm)
