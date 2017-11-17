from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, ones_like
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric


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
                 vocabulary: Vocabulary,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None) -> None:
        """
        Parameters
        ----------
        vocabulary : ``Vocabulary``, required.
            A vocabulary containing the tag namespace.
        tag_namespace : str, required.
            This metric assumes that a BIO format is used in which the
            labels are of the format: ["B-LABEL", "I-LABEL"].
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:

             ``ignore_classes=["V"]``
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.

            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]

            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        """
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes = ignore_classes or []

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 prediction_map: Optional[torch.Tensor] = None):
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
        prediction_map: ``torch.Tensor``, optional (default = None).
            A tensor of size (batch_size, num_classes) which provides a mapping from the index of predictions
            to the indices of the label vocabulary. If provided, the output label at each timestep will be
            ``vocabulary.get_index_to_token_vocabulary(prediction_map[batch, argmax(predictions[batch, t]))``,
            rather than simply ``vocabulary.get_index_to_token_vocabulary(argmax(predictions[batch, t]))``.
            This is useful in cases where each Instance in the dataset is associated with a different possible
            subset of labels from a large label-space (IE FrameNet, where each frame has a different set of
            possible roles associated with it).
        """
        if mask is None:
            mask = ones_like(gold_labels)
        # Get the data from the Variables.
        predictions, gold_labels, mask, prediction_map = self.unwrap_to_tensors(predictions,
                                                                                gold_labels,
                                                                                mask, prediction_map)

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to SpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(num_classes))

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        argmax_predictions = predictions.max(-1)[1]

        if prediction_map is not None:
            argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
            gold_labels = torch.gather(prediction_map, 1, gold_labels.long())

        argmax_predictions = argmax_predictions.float()

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            sequence_prediction = argmax_predictions[i, :]
            sequence_gold_label = gold_labels[i, :]
            length = sequence_lengths[i]

            if length == 0:
                # It is possible to call this metric with sequences which are
                # completely padded. These contribute nothing, so we skip these rows.
                continue
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
            if bio_tag == "O" or conll_tag in self._ignore_classes:
                # The span has ended.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                active_conll_tag = None
                # We don't care about tags we are
                # told to ignore, so we do nothing.
                continue
            elif bio_tag == "U":
                # The U tag is used to indicate a span of length 1,
                # so if there's an active tag we end it, and then
                # we add a "length 0" tag.
                if active_conll_tag:
                    spans.add(((span_start, span_end), active_conll_tag))
                spans.add(((index, index), conll_tag))
                active_conll_tag = None
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
        all_tags: Set[str] = set()
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
