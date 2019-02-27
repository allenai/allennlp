from typing import Dict

import torch
from allennlp.common import Params, Registrable

class LossWeighter(Registrable):
    """
    This class is an abstract class for loss weighters.
    Whenever the loss function is composed of more then a single term weighting is probable.
    Use children of this class for different constant/annealed weights
    """
    def __init__(self, initial_weight: float) -> None:
        self._weight = initial_weight

    def next(self) -> float:
        self.step()
        return self.get()

    def get(self) -> float:
        return self._weight

    def step(self):
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> Dict[str, 'LossWeighter']:
        # pylint: disable=arguments-differ
        weighters = {}
        for param_name in params:
            curr_params = params.get(param_name)
            weighter_type = curr_params.pop("type", "constant_weight")
            weighter = LossWeighter.by_name(weighter_type)(**curr_params.as_dict())
            weighters[param_name] = weighter

        return weighters

class ConstantWeight(LossWeighter):
    """
    This class is for a constant weight scalar.
    """
    def step(self) -> None:
        pass

class _Annealer(LossWeighter):
    def __init__(self, min_weight: float, max_weight: float, warmup: int, num_iter_to_max: int) -> None:
        """
        This class is an abstract class for annealing loss weighters.
        """
        super(_Annealer, self).__init__(min_weight)
        self.iteration = 0
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.warmup = warmup
        self.num_iter_to_max = num_iter_to_max

    def step(self):
        self._take_step()
        self.iteration += 1

    def _take_step(self):
        raise NotImplementedError

class LinearAnnealer(_Annealer):
    """
    This class anneals weights linearly.
    """
    def _take_step(self):
        if self.iteration < self.warmup:
            self._weight = self.min_weight
        elif self.num_iter_to_max is not None and self.iteration > self.num_iter_to_max:
            self._weight = self.max_weight
        else:
            self._weight = self.min_weight + (self.max_weight-self.min_weight)*(self.iteration-self.warmup)/ \
                                             (self.num_iter_to_max-self.warmup)

class SigmoidAnnealer(_Annealer):
    """
    This class anneals weights in a sigmoid fashion.
    """
    def __init__(self, min_weight: float,
                 max_weight: float,
                 warmup: int,
                 num_iter_to_max: int,
                 slope: float) -> None:
        super(SigmoidAnnealer, self).__init__(min_weight, max_weight, warmup, num_iter_to_max)
        self.slope = slope

    def _take_step(self):
        shifted_max = self.max_weight - self.min_weight
        middle_point = (self.warmup + self.num_iter_to_max)/2
        res = shifted_max * torch.sigmoid(torch.Tensor([self.slope*(self.iteration-middle_point)])) + \
                            self.min_weight
        self._weight = round(res.item(), 2)

Registrable._registry[LossWeighter] = {  # pylint: disable=protected-access
        "constant_weight": ConstantWeight,
        "linear_annealer": LinearAnnealer,
        "sigmoid_annealer": SigmoidAnnealer
}
