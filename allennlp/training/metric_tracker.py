from typing import Optional, Dict, Any, List, Union

from allennlp.common.checks import ConfigurationError


class MetricTracker:
    """
    This class tracks a metric during training for the dual purposes of early stopping
    and for knowing whether the current value is the best so far. It mimics the PyTorch
    `state_dict` / `load_state_dict` interface, so that it can be checkpointed along with
    your model and optimizer.

    Some metrics improve by increasing; others by decreasing. You can provide a
    `metric_name` that starts with "+" to indicate an increasing metric, or "-"
    to indicate a decreasing metric.

    # Parameters

    metric_name : `Union[str, List[str]]`
        Specifies the metric or metrics to track. Metric names have to start with
        "+" for increasing metrics or "-" for decreasing ones. If you specify more
        than one, it tracks the sum of the increasing metrics metrics minus the sum
        of the decreasing metrics.
    patience : `int`, optional (default = `None`)
        If provided, then `should_stop_early()` returns True if we go this
        many epochs without seeing a new best value.
    """

    def __init__(
        self,
        metric_name: Union[str, List[str]],
        patience: Optional[int] = None,
    ) -> None:
        self._patience = patience
        self._best_so_far: Optional[float] = None
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self._epoch_number = 0
        self.best_epoch: Optional[int] = None
        self.best_epoch_metrics: Dict[str, float] = {}

        if isinstance(metric_name, str):
            metric_name = [metric_name]
        self.tracked_metrics = []
        for name in metric_name:
            if name.startswith("+"):
                self.tracked_metrics.append((1.0, name[1:]))
            elif name.startswith("-"):
                self.tracked_metrics.append((-1.0, name[1:]))
            else:
                raise ConfigurationError("metric_name must start with + or -")

    def clear(self) -> None:
        """
        Clears out the tracked metrics, but keeps the patience
        """
        self._best_so_far = None
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = True
        self._epoch_number = 0
        self.best_epoch = None
        self.best_epoch_metrics.clear()

    def state_dict(self) -> Dict[str, Any]:
        """
        A `Trainer` can use this to serialize the state of the metric tracker.
        """
        return {
            "best_so_far": self._best_so_far,
            "epochs_with_no_improvement": self._epochs_with_no_improvement,
            "is_best_so_far": self._is_best_so_far,
            "epoch_number": self._epoch_number,
            "best_epoch": self.best_epoch,
            "best_epoch_metrics": self.best_epoch_metrics,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        A `Trainer` can use this to hydrate a metric tracker from a serialized state.
        """
        self._best_so_far = state_dict["best_so_far"]
        self._epochs_with_no_improvement = state_dict["epochs_with_no_improvement"]
        self._is_best_so_far = state_dict["is_best_so_far"]
        self._epoch_number = state_dict["epoch_number"]
        self.best_epoch = state_dict["best_epoch"]

        # Even though we don't promise backwards compatibility for the --recover flag,
        # it's particularly easy and harmless to provide it here, so we do it.
        self.best_epoch_metrics = state_dict.get("best_epoch_metrics", {})

    def add_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Record a new value of the metric and update the various things that depend on it.
        """
        try:
            combined_score = sum(
                factor * metrics[metric_name] for factor, metric_name in self.tracked_metrics
            )
        except KeyError as e:
            raise ConfigurationError(
                f"You configured the trainer to use the {e.args[0]}"
                "metric for early stopping, but the model did not produce that metric."
            )

        new_best = (self._best_so_far is None) or (combined_score > self._best_so_far)

        if new_best:
            self._best_so_far = combined_score
            self._epochs_with_no_improvement = 0
            self._is_best_so_far = True
            self.best_epoch = self._epoch_number
        else:
            self._epochs_with_no_improvement += 1
            self._is_best_so_far = False
        self._epoch_number += 1

    def is_best_so_far(self) -> bool:
        """
        Returns true if the most recent value of the metric is the best so far.
        """
        return self._is_best_so_far

    def should_stop_early(self) -> bool:
        """
        Returns true if improvement has stopped for long enough.
        """
        if self._patience is None:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience
