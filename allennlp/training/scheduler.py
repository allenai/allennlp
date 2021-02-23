from typing import Dict, Any

import torch


class Scheduler:
    """
    A `Scheduler` is a generalization of PyTorch learning rate schedulers.

    A scheduler can be used to update any field in an optimizer's parameter groups,
    not just the learning rate.

    During training using the AllenNLP `Trainer`, this is the API and calling
    sequence for `step` and `step_batch`::

       scheduler = ... # creates scheduler

       batch_num_total = 0
       for epoch in range(num_epochs):
           for batch in batchs_in_epoch:
               # compute loss, update parameters with current learning rates
               # call step_batch AFTER updating parameters
               batch_num_total += 1
               scheduler.step_batch(batch_num_total)
           # call step() at the END of each epoch
           scheduler.step(validation_metrics, epoch)
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, param_group_field: str, last_epoch: int = -1
    ) -> None:
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
                    raise KeyError(
                        f"{self._initial_param_group_field} missing from param_groups[{i}]"
                    )
        self.base_values = [
            group[self._initial_param_group_field] for group in self.optimizer.param_groups
        ]
        self.last_epoch = last_epoch

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the scheduler as a `dict`.
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the schedulers state.

        # Parameters

        state_dict : `Dict[str, Any]`
            Scheduler state. Should be an object returned from a call to `state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_values(self):
        raise NotImplementedError

    def step(self, metric: float = None) -> None:
        self.last_epoch += 1
        self.metric = metric
        for param_group, value in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = value

    def step_batch(self, batch_num_total: int = None) -> None:
        """
        By default, a scheduler is assumed to only update every epoch, not every batch.
        So this does nothing unless it's overriden.
        """

        return
