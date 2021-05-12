"""
This module contains tools to:

1. measure the fairness of models according to multiple definitions of fairness
2. measure bias amplification
3. debias embeddings during training time and post-processing
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
