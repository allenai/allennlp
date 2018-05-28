from typing import Dict

from allennlp.models import Model


class ModelMetricsFacade:
    """
    Wraps around a model and provides a single point to query to get all the metrics of a model
    along with its computed loss so far.

    This class is also responsible for keeping track of the values i.e. loss and number of batches per epoch
    """

    def __init__(self,
                 model: Model) -> None:
        self._model = model
        self.batches_this_epoch = 0
        self.train_loss = 0.0

    def compute_for_running_epoch(self) -> Dict[str, float]:
        """
        Compute the current loss for this epoch
        """
        return self._compute_metrics_and_include_loss(self._model.get_metrics(),
                                                      self.batches_this_epoch,
                                                      self.train_loss)

    def compute_for_completed_epoch(self) -> Dict[str, float]:
        """
        gets the loss for this epoch and resets metrics of underlying model for next epoch
        """
        return self._compute_metrics_and_include_loss(self._model.get_metrics(reset=True),
                                                      self.batches_this_epoch,
                                                      self.train_loss)

    def new_batch(self):
        self.batches_this_epoch += 1

    def increase_loss(self, loss: float):
        self.train_loss += loss

    @classmethod
    def _compute_metrics_and_include_loss(cls,
                                          metrics: Dict[str, float],
                                          batches_this_epoch: int,
                                          train_loss: float) -> Dict[str, float]:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``num_batches`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        metrics = metrics
        metrics["loss"] = float(train_loss / batches_this_epoch) if batches_this_epoch > 0 else 0.0
        return metrics
