from typing import Dict, Any

import torch


class Scheduler:
    """
    A ``Scheduler`` is a generalization of PyTorch learning rate schedulers.

    A scheduler can be used to update any field in an optimizer's parameter groups,
    not just the learning rate.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if last_epoch == -1:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the scheduler as a ``dict``.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the schedulers state.

        Parameters
        ----------
        state_dict : ``Dict[str, Any]``
            Scheduler state. Should be an object returned from a call to ``state_dict``.
        """
        self.__dict__.update(state_dict)

    def step(self, metric: float, epoch: int = None) -> None:
        raise NotImplementedError

    def step_batch(self, batch_num_total: Optional[int]) -> None:
        """
        By default, a scheduler is assumed to only update every epoch, not every batch.
        So this does nothing unless it's overriden.
        """
        # pylint: disable=unused-argument,no-self-use
        return
