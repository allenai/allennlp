"""
This module contains tools to:

1. measure the fairness of models according to multiple definitions of fairness
2. measure bias amplification
3. mitigate bias in static and contextualized embeddings during training time and
post-processing
"""

from allennlp.fairness.fairness_metrics import Independence, Separation, Sufficiency
from allennlp.fairness.bias_metrics import (
    WordEmbeddingAssociationTest,
    EmbeddingCoherenceTest,
    NaturalLanguageInference,
    AssociationWithoutGroundTruth,
)
from allennlp.fairness.bias_direction import (
    PCABiasDirection,
    PairedPCABiasDirection,
    ClassificationNormalBiasDirection,
    TwoMeansBiasDirection,
)
from allennlp.fairness.bias_mitigators import (
    LinearBiasMitigator,
    HardBiasMitigator,
    INLPBiasMitigator,
    OSCaRBiasMitigator,
)
from allennlp.fairness.bias_utils import load_words, load_word_pairs
from allennlp.fairness.bias_mitigator_applicator import BiasMitigatorApplicator
from allennlp.fairness.bias_mitigator_wrappers import (
    HardBiasMitigatorWrapper,
    LinearBiasMitigatorWrapper,
    INLPBiasMitigatorWrapper,
    OSCaRBiasMitigatorWrapper,
)
from allennlp.fairness.bias_direction_wrappers import (
    PCABiasDirectionWrapper,
    PairedPCABiasDirectionWrapper,
    TwoMeansBiasDirectionWrapper,
    ClassificationNormalBiasDirectionWrapper,
)
