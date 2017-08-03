
import torch
from allennlp.common.registrable import Registrable


class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """

    def __call__(self, *args, **kwargs):

        raise NotImplementedError

    def finalise_and_reset(self):
        raise NotImplementedError


@Metric.register("categorical_accuracy")
class CategoricalAccuracy(Metric):

    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor):

        # Top K indexes of the predictions.
        top_k = predictions.topk(self.top_k, 1)[1]
        # expand labels to be the same shape.
        true_k = gold_labels.view(gold_labels.size()[0], 1).expand_as(top_k)
        self.correct_count += top_k.equal(true_k).float().sum().data[0]
        self.total_count += predictions.size(0)

    def finalise_and_reset(self):
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        self.correct_count = 0.0
        self.total_count = 0.0
        return {"accuracy": accuracy}

@Metric.register("f1")
class F1Measure(Metric):

    def __init__(self):

        self.true_positives = 0.0
        self.true_negatives = 0.0
        self.false_positives = 0.0
        self.false_negatives = 0.0
        self.total_counts = 0.0

    def __call__(self, predictions, gold_labels):

