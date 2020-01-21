from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging
import os
import tempfile
import subprocess
import shutil

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from allennlp.models.srl_util import write_conll_formatted_tags_to_file

logger = logging.getLogger(__name__)

DEFAULT_SRL_EVAL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "tools", "srl-eval.pl"
    )
)


@Metric.register("srl_eval")
class SrlEvalScorer(Metric):
    """
    This class uses the external srl-eval.pl script for computing the CoNLL SRL metrics.

    AllenNLP contains the srl-eval.pl script, but you will need perl 5.x.

    Note that this metric reads and writes from disk quite a bit. In particular, it
    writes and subsequently reads two files per __call__, which is typically invoked
    once per batch. You probably don't want to include it in your training loop;
    instead, you should calculate this on a validation set only.

    # Parameters

    srl_eval_path : `str`, optional.
        The path to the srl-eval.pl script.
    ignore_classes : `List[str]`, optional (default=`None`).
        A list of classes to ignore.
    """

    def __init__(
        self, srl_eval_path: str = DEFAULT_SRL_EVAL_PATH, ignore_classes: List[str] = None
    ) -> None:
        self._srl_eval_path = srl_eval_path
        self._ignore_classes = set(ignore_classes)
        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    @overrides
    def __call__(
        self,  # type: ignore
        batch_verb_indices: List[Optional[int]],
        batch_sentences: List[List[str]],
        batch_conll_formatted_predicted_tags: List[List[str]],
        batch_conll_formatted_gold_tags: List[List[str]],
    ) -> None:

        """
        # Parameters

        batch_verb_indices : `List[Optional[int]]`, required.
            The indices of the verbal predicate in the sentences which
            the gold labels are the arguments for, or None if the sentence
            contains no verbal predicate.
        batch_sentences : `List[List[str]]`, required.
            The word tokens for each instance in the batch.
        batch_conll_formatted_predicted_tags : `List[List[str]]`, required.
            A list of predicted CoNLL-formatted SRL tags (itself a list) to compute score for.
            Use allennlp.models.semantic_role_labeler.convert_bio_tags_to_conll_format
            to convert from BIO to CoNLL format before passing the tags into the metric,
            if applicable.
        batch_conll_formatted_gold_tags : `List[List[str]]`, required.
            A list of gold CoNLL-formatted SRL tags (itself a list) to use as a reference.
            Use allennlp.models.semantic_role_labeler.convert_bio_tags_to_conll_format
            to convert from BIO to CoNLL format before passing the
            tags into the metric, if applicable.
        """
        if not os.path.exists(self._srl_eval_path):
            raise ConfigurationError(f"srl-eval.pl not found at {self._srl_eval_path}.")
        tempdir = tempfile.mkdtemp()
        gold_path = os.path.join(tempdir, "gold.txt")
        predicted_path = os.path.join(tempdir, "predicted.txt")

        with open(predicted_path, "w", encoding="utf-8") as predicted_file, open(
            gold_path, "w", encoding="utf-8"
        ) as gold_file:
            for verb_index, sentence, predicted_tag_sequence, gold_tag_sequence in zip(
                batch_verb_indices,
                batch_sentences,
                batch_conll_formatted_predicted_tags,
                batch_conll_formatted_gold_tags,
            ):
                write_conll_formatted_tags_to_file(
                    predicted_file,
                    gold_file,
                    verb_index,
                    sentence,
                    predicted_tag_sequence,
                    gold_tag_sequence,
                )
        perl_script_command = ["perl", self._srl_eval_path, gold_path, predicted_path]
        completed_process = subprocess.run(
            perl_script_command, stdout=subprocess.PIPE, universal_newlines=True, check=True
        )
        for line in completed_process.stdout.split("\n"):
            stripped = line.strip().split()
            if len(stripped) == 7:
                tag = stripped[0]
                # Overall metrics are calculated in get_metric, skip them here.
                if tag == "Overall" or tag in self._ignore_classes:
                    continue
                # This line contains results for a span
                num_correct = int(stripped[1])
                num_excess = int(stripped[2])
                num_missed = int(stripped[3])
                self._true_positives[tag] += num_correct
                self._false_positives[tag] += num_excess
                self._false_negatives[tag] += num_missed
        shutil.rmtree(tempdir)

    def get_metric(self, reset: bool = False):
        """
        # Returns

        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an `overall` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            if tag == "overall":
                raise ValueError(
                    "'overall' is disallowed as a tag type, "
                    "rename the tag type to something else if necessary."
                )
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )
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
        f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
