from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

from overrides import overrides
import torch
from torch.autograd import Variable

from allennlp.common.registrable import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.common.params import Params


class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor]):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params):
        metric_type = params.pop_choice("type", cls.list_available())
        return cls.by_name(metric_type)(**params.as_dict())  # type: ignore


@Metric.register("categorical_accuracy")
class CategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    """
    def __init__(self, top_k: int = 1) -> None:
        self._top_k = top_k
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # If you actually passed in Variables here instead of Tensors, this will be a huge memory
        # leak, because it will prevent garbage collection for the computation graph.  We'll ensure
        # that we're using tensors here first.
        if isinstance(predictions, Variable):
            predictions = predictions.data
        if isinstance(gold_labels, Variable):
            gold_labels = gold_labels.data
        if isinstance(mask, Variable):
            mask = mask.data

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        # Top K indexes of the predictions.
        top_k = predictions.topk(self._top_k, -1)[1]

        # This is of shape (batch_size, ..., top_k).
        correct = top_k.eq(gold_labels.long().unsqueeze(-1)).float()
        count = torch.ones(gold_labels.size())
        if mask is not None:
            correct *= mask.unsqueeze(-1)
            count *= mask
        self.correct_count += correct.sum()
        self.total_count += count.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = float(self.correct_count) / float(self.total_count)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


@Metric.register("boolean_accuracy")
class BooleanAccuracy(Metric):
    """
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.  This
    is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """
    def __init__(self) -> None:
        self._correct_count = 0.
        self._total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        # If you actually passed in Variables here instead of Tensors, this will be a huge memory
        # leak, because it will prevent garbage collection for the computation graph.  We'll ensure
        # that we're using tensors here first.
        if isinstance(predictions, Variable):
            predictions = predictions.data
        if isinstance(gold_labels, Variable):
            gold_labels = gold_labels.data
        if isinstance(mask, Variable):
            mask = mask.data

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)

        # The .prod() here is functioning as a logical and.
        correct = predictions.eq(gold_labels).prod(dim=1).float()
        count = torch.ones(gold_labels.size(0))
        self._correct_count += correct.sum()
        self._total_count += count.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = float(self._correct_count) / float(self._total_count)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self._correct_count = 0.0
        self._total_count = 0.0


@Metric.register("f1")
class F1Measure(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, positive_label: int) -> None:
        self._positive_label = positive_label
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # If you actually passed in Variables here instead of Tensors, this will be a huge memory
        # leak, because it will prevent garbage collection for the computation graph.  We'll ensure
        # that we're using tensors here first.
        if isinstance(predictions, Variable):
            predictions = predictions.data
        if isinstance(gold_labels, Variable):
            gold_labels = gold_labels.data
        if isinstance(mask, Variable):
            mask = mask.data

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to F1Measure contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        if mask is None:
            mask = torch.ones(gold_labels.size())
        mask = mask.float()
        gold_labels = gold_labels.float()
        positive_label_mask = gold_labels.eq(self._positive_label).float()
        negative_label_mask = 1.0 - positive_label_mask

        argmax_predictions = predictions.topk(1, -1)[1].float().squeeze(-1)

        # True Negatives: correct non-positive predictions.
        correct_null_predictions = (argmax_predictions !=
                                    self._positive_label).float() * negative_label_mask
        self._true_negatives += (correct_null_predictions.float() * mask).sum()

        # True Positives: correct positively labeled predictions.
        correct_non_null_predictions = (argmax_predictions ==
                                        self._positive_label).float() * positive_label_mask
        self._true_positives += (correct_non_null_predictions * mask).sum()

        # False Negatives: incorrect negatively labeled predictions.
        incorrect_null_predictions = (argmax_predictions !=
                                      self._positive_label).float() * positive_label_mask
        self._false_negatives += (incorrect_null_predictions * mask).sum()

        # False Positives: incorrect positively labeled predictions
        incorrect_non_null_predictions = (argmax_predictions ==
                                          self._positive_label).float() * negative_label_mask
        self._false_positives += (incorrect_non_null_predictions * mask).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0


@Metric.register("span_f1")
class SpanBasedF1Measure(Metric):
    """
    The Conll SRL metrics are based on exact span matching. This metric
    implements span-based precision and recall metrics for a BIO tagging
    scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2005
    data - particularly, it does not consider continuations or reference spans
    as constituents of the original span. However, it is a close proxy, which
    can be helpful for judging model peformance during training.
    """
    def __init__(self,
                 label_vocabulary: Dict[int, str],
                 ignore_classes: List[str] = None) -> None:
        """
        Parameters
        ----------
        label_vocabulary : Dict[int, str], required.
            A mapping from integer class labels to string labels. This metric
            assumes that a BIO format is used in which the labels are of the format:
            ["B-LABEL", "I-LABEL"].
        ignore_classes : List[str], optional.
            Non-BIO labeled classes which will be ignored when computing span
            metrics. This is helpful for instance, to avoid computing
            metrics for "O", and "V" tags in a BIO tagging scheme
            for semantic role labelling, which are typically not included.
            Note that these labels do not have their prepended BIO information,
            they are only the raw tag.
        """
        self._label_vocabulary = label_vocabulary
        self._ignore_classes = ignore_classes or []

        # These will hold per label span counts.
        self._true_positives = defaultdict(int)  # type: Dict[str, int]
        self._false_positives = defaultdict(int)  # type: Dict[str, int]
        self._false_negatives = defaultdict(int)  # type: Dict[str, int]

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        if mask is None:
            mask = torch.ones(gold_labels.size())
        # If you actually passed in Variables here instead of Tensors, this will be a huge memory
        # leak, because it will prevent garbage collection for the computation graph.  We'll ensure
        # that we're using tensors here first.
        if isinstance(predictions, Variable):
            predictions = predictions.data.cpu()
        if isinstance(gold_labels, Variable):
            gold_labels = gold_labels.data.cpu()
        if isinstance(mask, Variable):
            mask = mask.data.cpu()

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to SpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(num_classes))

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.max(-1)[1].float().squeeze(-1)

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]
            prediction_spans = self._extract_spans(sequence_prediction[:length].tolist())
            gold_spans = self._extract_spans(sequence_gold_label[:length].tolist())

            for span in prediction_spans:
                if span in gold_spans:
                    self._true_positives[span[1]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[1]] += 1
            # These spans weren't predicted.
            for span in gold_spans:
                self._false_negatives[span[1]] += 1

    def _extract_spans(self, tag_sequence: List[int]) -> Set[Tuple[Tuple[int, int], str]]:
        """
        Given an integer tag sequence corresponding to BIO tags, extracts spans.
        Spans are inclusive and can be of zero length, representing a single word span.
        Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
        as otherwise it is possible to get a perfect precision score whilst still predicting
        ill-formed spans in addition to the correct spans.

        Parameters
        ----------
        tag_sequence : List[int], required.
            The integer class labels for a sequence.

        Returns
        -------
        spans : Set[Tuple[Tuple[int, int], str]]
            The typed, extracted spans from the sequence, in the format ((span_start, span_end), label).
            Note that the label `does not` contain any BIO tag prefixes.
        """
        spans = set()
        span_start = 0
        span_end = 0
        active_conll_tag = None
        for index, integer_tag in enumerate(tag_sequence):
            # Actual BIO tag.
            string_tag = self._label_vocabulary[integer_tag]
            bio_tag = string_tag[0]
            conll_tag = string_tag[2:]
            if conll_tag in self._ignore_classes:
                # The span has ended.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                active_conll_tag = None
                # We don't care about tags we are
                # told to ignore, so we do nothing.
                continue
            elif bio_tag == "B":
                # We are entering a new span; reset indices
                # and active tag to new span.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
            elif bio_tag == "I" and conll_tag == active_conll_tag:
                # We're inside a span.
                span_end += 1
            else:
                # This is the case the bio label is an "I", but either:
                # 1) the span hasn't started - i.e. an ill formed span.
                # 2) The span is an I tag for a different conll annotation.
                # We'll process the previous span if it exists, but also
                # include this span. This is important, because otherwise,
                # a model may get a perfect F1 score whilst still including
                # false positive ill-formed spans.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                active_conll_tag = conll_tag
                span_start = index
                span_end = index
        # Last token might have been a part of a valid span.
        if active_conll_tag:
            spans.add(((span_start, span_end), active_conll_tag))
        return spans

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags = set()  # type: Set[str]
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
