from typing import Dict, Iterable
from allennlp.models.model import Model


class ExponentialMovingAverage:
    def __init__(self, model: Model, decay: float = 0.9999) -> None:
        """
        Create shadow variables and maintain exponential moving average for model parameters.

        Parameters
        ----------
        model: ``Model``, required.
            An AllenNLP model to be optimized.
        decay: ``float``, optional, (default=0.9999).
            The decay rate for each step. Note that a dynamic decay rate might be used if
            `num_updates` is provided when applying the exponential moving average.
        """
        self.decay = decay
        self._average_values: Dict = {}
        self._backup_values: Dict = {}
        self._model = model
        for name, param in model.named_parameters():
            self._average_values[name] = param.data.clone()
            self._backup_values[name] = param.data.clone()

    def apply(self, num_updates: int = None, named_parameters: Iterable = None) -> None:
        """
        Apply exponential moving average to `named_parameters` if specified,
        or we will apply this to all the trainable parameters of the model.

        The optional `num_updates` parameter allows one to tweak the decay rate
        dynamically. If passed, the actual decay rate used is:

            `min(decay, (1 + num_updates) / (10 + num_updates))`

        """
        if num_updates is not None:
            decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        else:
            decay = self.decay
        if named_parameters is None:
            named_parameters = self._model.named_parameters()
        for name, param in named_parameters:
            new_average_value = (1.0 - decay) * param.data + decay * self._average_values[name]
            self._average_values[name] = new_average_value.clone()

    def assign_average_value(self, named_parameters=None) -> None:
        """
        Assign the exponential moving average value to the parameters
        """
        if named_parameters is None:
            named_parameters = self._model.named_parameters()
        for name, param in named_parameters:
            self._backup_values[name] = param.data.clone()
            param.data = self._average_values[name]

    def restore(self, named_parameters=None) -> None:
        """
        Restore the original values of each parameter
        """
        if named_parameters is None:
            named_parameters = self._model.named_parameters()
        for name, param in named_parameters:
            param.data = self._backup_values[name].clone()
