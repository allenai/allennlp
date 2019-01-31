import torch

from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.training.scheduler import Scheduler


class MomentumScheduler(Scheduler, Registrable):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 last_epoch: int = -1) -> None:
        super().__init__(optimizer, "momentum", last_epoch)

    def get_values(self) -> None:
        raise NotImplementedError

    # Requires custom from_params so we can pass the optimizer.
    @classmethod
    def from_params(cls, optimizer: torch.optim.Optimizer, params: Params):  # type: ignore
        # pylint: disable=arguments-differ
        scheduler_type = params.pop_choice("type", MomentumScheduler.list_available())
        scheduler = MomentumScheduler.by_name(scheduler_type)(optimizer, **params.as_dict())
        return scheduler
