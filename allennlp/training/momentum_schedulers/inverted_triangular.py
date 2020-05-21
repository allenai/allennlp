import torch

from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler


@MomentumScheduler.register("inverted_triangular")
class InvertedTriangular(MomentumScheduler):
    """
    Adjust momentum during training according to an inverted triangle-like schedule.

    The momentum starts off high, then decreases linearly for `cool_down` epochs,
    until reaching `1 / ratio` th of the original value. Then the momentum increases
    linearly for `warm_up` epochs until reaching its original value again. If there
    are still more epochs left over to train, the momentum will stay flat at the original
    value.

    Registered as a `MomentumScheduler` with name "inverted_triangular".  The "optimizer" argument
    does not get an entry in a configuration file for the object.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cool_down: int,
        warm_up: int,
        ratio: int = 10,
        last_epoch: int = -1,
    ) -> None:
        self.cool_down = cool_down
        self.warm_up = warm_up
        self.ratio = ratio
        super().__init__(optimizer, last_epoch)

    def get_values(self):
        step = self.last_epoch + 1
        if step <= self.cool_down:
            values = [m - (m - m / self.ratio) * (step / self.cool_down) for m in self.base_values]
        elif step <= self.cool_down + self.warm_up:
            values = [
                (m / self.ratio) + (m - m / self.ratio) * (step - self.cool_down) / self.warm_up
                for m in self.base_values
            ]
        else:
            values = self.base_values

        return values
