
import torch
import torch.optim.lr_scheduler
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable


class LearningRateScheduler(Registrable):
    """
    This class just allows us to implement ``Registerable`` for Pytorch :class:`LRSchedulers`.
    """
    @classmethod
    def from_params(cls, optimizer: torch.optim.Optimizer, params: Params):
        scheduler = params.pop_choice("type", LearningRateScheduler.list_available())
        return LearningRateScheduler.by_name(scheduler)(optimizer, **params.as_dict())  # type: ignore


# We just use the Pytorch LRSchedulers, so here we force them into
# Registry._registry so we can build them from params.
Registrable._registry[LearningRateScheduler] = {   # pylint: disable=protected-access
        "step": torch.optim.lr_scheduler.StepLR,
        "multi_step": torch.optim.lr_scheduler.MultiStepLR,
        "exponential": torch.optim.lr_scheduler.ExponentialLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau
}
