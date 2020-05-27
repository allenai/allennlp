from typing import Iterable, Tuple, Optional

import torch

from allennlp.common.registrable import Registrable

NamedParameter = Tuple[str, torch.Tensor]


class MovingAverage(Registrable):
    """
    Tracks a moving average of model parameters.
    """

    default_implementation = "exponential"

    def __init__(self, parameters: Iterable[NamedParameter]) -> None:
        self._parameters = list(parameters)
        self._shadows = {name: parameter.data.clone() for name, parameter in self._parameters}
        self._backups = {name: parameter.data.clone() for name, parameter in self._parameters}

    def apply(self, num_updates: Optional[int] = None):
        """
        Update the moving averages based on the latest values of the parameters.
        """
        raise NotImplementedError

    def assign_average_value(self) -> None:
        """
        Replace all the parameter values with the averages.
        Save the current parameter values to restore later.
        """
        for name, parameter in self._parameters:
            self._backups[name].copy_(parameter.data)
            parameter.data.copy_(self._shadows[name])

    def restore(self) -> None:
        """
        Restore the backed-up (non-average) parameter values.
        """
        for name, parameter in self._parameters:
            parameter.data.copy_(self._backups[name])


@MovingAverage.register("exponential")
class ExponentialMovingAverage(MovingAverage):
    """
    Create shadow variables and maintain exponential moving average for model parameters.

    Registered as a `MovingAverage` with name "exponential".

    # Parameters

    parameters : `Iterable[Tuple[str, Parameter]]`, required
        The parameters whose averages we'll be tracking.

        In a typical AllenNLP configuration file, this argument does not get an entry under the
        "moving_average", it gets passed in separately.
    decay : `float`, optional (default = `0.9999`)
        The decay rate that will be used if `num_updates` is not passed
        (and that will be used as an upper bound if `num_updates` is passed).
    numerator : `float`, optional (default = `1.0`)
        The numerator used to compute the decay rate if `num_updates` is passed.
    denominator : `float`, optional (default = `10.0`)
        The denominator used to compute the decay rate if `num_updates` is passed.
    """

    def __init__(
        self,
        parameters: Iterable[NamedParameter],
        decay: float = 0.9999,
        numerator: float = 1.0,
        denominator: float = 10.0,
    ) -> None:
        super().__init__(parameters)
        self._decay = decay
        self._numerator = numerator
        self._denominator = denominator

    def apply(self, num_updates: Optional[int] = None) -> None:
        """
        Apply exponential moving average to `named_parameters` if specified,
        or we will apply this to all the trainable parameters of the model.

        The optional `num_updates` parameter allows one to tweak the decay rate
        dynamically. If passed, the actual decay rate used is:

            `min(decay, (numerator + num_updates) / (denominator + num_updates))`

        (This logic is based on the Tensorflow exponential moving average
         <https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage>)
        """
        if num_updates is not None:
            decay = min(
                self._decay, (self._numerator + num_updates) / (self._denominator + num_updates)
            )
        else:
            decay = self._decay

        for name, parameter in self._parameters:
            self._shadows[name].mul_(decay).add_((1 - decay) * parameter.data)
