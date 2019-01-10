from typing import Optional, Iterable

from allennlp.common.checks import ConfigurationError

class MetricTracker:
    """
    This class tracks a metric for the dual purposes of early stopping
    and for knowing whether the current value is the best so far.

    Parameters
    ----------
    patience : int, optional (default = None)
        If provided, then `should_stop_early()` returns True if we go this
        many epochs without seeing a new best value.
    metric_name : str, optional (default = None)
        If provided, it's used to infer whether we expect the metric values to
        increase (if it starts with "+") or decrease (if it starts with "-").
        It's an error if it doesn't start with one of those. If it's not provided,
        you should specify ``should_decrease`` instead.
    should_decrease : str, optional (default = None)
        If ``metric_name`` isn't provided (in which case we can't infer ``should_decrease``),
        then you have to specify it here.
    """
    def __init__(self,
                 patience: Optional[int] = None,
                 metric_name: str = None,
                 should_decrease: bool = None) -> None:
        self._best_so_far: float = None
        self._patience = patience
        self._epochs_with_no_improvement = 0

        self.is_best_so_far = True

        # If the metric name starts with "+", we want it to increase.
        # If the metric name starts with "-", we want it to decrease.
        # We also allow you to not specify a metric name and just set `should_decrease` directly.
        if should_decrease is None and metric_name is None:
            raise ConfigurationError("must specify either `should_decrease` or `metric_name` (but not both)")
        elif should_decrease is not None and metric_name is not None:
            raise ConfigurationError("must specify either `should_decrease` or `metric_name` (but not both)")
        elif metric_name is not None:
            if metric_name[0] == "-":
                self._should_decrease = True
            elif metric_name[0] == "+":
                self._should_decrease = False
            else:
                raise ConfigurationError("metric_name must start with + or -")
        else:
            self._should_decrease = should_decrease

    def clear(self) -> None:
        """
        Clears out the tracked metrics, but keeps the patience and should_decrease settings.
        """
        self._best_so_far = None
        self._epochs_with_no_improvement = 0
        self.is_best_so_far = True

    def state_dict(self):
        """
        A ``Trainer`` can use this to serialize the state of the metric tracker.
        """
        return {
                "best_so_far": self._best_so_far,
                "patience": self._patience,
                "epochs_with_no_improvement": self._epochs_with_no_improvement,
                "is_best_so_far": self.is_best_so_far,
                "should_decrease": self._should_decrease
        }

    def load_state_dict(self, state_dict) -> None:
        """
        A ``Trainer`` can use this to hydrate a metric tracker from a serialized state.
        """
        self._best_so_far = state_dict["best_so_far"]
        self._patience = state_dict["patience"]
        self._epochs_with_no_improvement = state_dict["epochs_with_no_improvement"]
        self.is_best_so_far = state_dict["is_best_so_far"]
        self._should_decrease = state_dict["should_decrease"]

    def add_metric(self, metric: float) -> None:
        """
        Record a new value of the metric and update the various things that depend on it.
        """
        new_best = ((self._best_so_far is None) or
                    (self._should_decrease and metric < self._best_so_far) or
                    (not self._should_decrease and metric > self._best_so_far))

        if new_best:
            self.is_best_so_far = True
            self._best_so_far = metric
            self._epochs_with_no_improvement = 0
        else:
            self.is_best_so_far = False
            self._epochs_with_no_improvement += 1

    def add_metrics(self, metrics: Iterable[float]) -> None:
        """
        Helper to add multiple metrics at once.
        """
        for metric in metrics:
            self.add_metric(metric)

    def should_stop_early(self) -> bool:
        """
        Returns true if improvement has stopped for long enough.
        """
        if self._patience is None:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience
