"""
A `~allennlp.training.metrics.metric.Metric` is some quantity or quantities
that can be accumulated during training or evaluation; for example,
accuracy or F1 score.
"""

from allennlp.training.metrics.attachment_scores import AttachmentScores
from allennlp.training.metrics.average import Average
from allennlp.training.metrics.boolean_accuracy import BooleanAccuracy
from allennlp.training.metrics.bleu import BLEU
from allennlp.training.metrics.rouge import ROUGE
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.training.metrics.covariance import Covariance
from allennlp.training.metrics.entropy import Entropy
from allennlp.training.metrics.evalb_bracketing_scorer import (
    EvalbBracketingScorer,
    DEFAULT_EVALB_DIR,
)
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.training.metrics.fbeta_multi_label_measure import (
    FBetaMultiLabelMeasure,
    F1MultiLabelMeasure,
)
from allennlp.training.metrics.f1_measure import F1Measure
from allennlp.training.metrics.mean_absolute_error import MeanAbsoluteError
from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.pearson_correlation import PearsonCorrelation
from allennlp.training.metrics.spearman_correlation import SpearmanCorrelation
from allennlp.training.metrics.perplexity import Perplexity
from allennlp.training.metrics.sequence_accuracy import SequenceAccuracy
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure
from allennlp.training.metrics.unigram_recall import UnigramRecall
from allennlp.training.metrics.auc import Auc
