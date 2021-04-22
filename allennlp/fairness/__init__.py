"""
This module contains tools to:

1. measure the fairness of models according to multiple definitions of fairness
2. measure bias amplification
3. debias embeddings during training time and post-processing
"""

from allennlp.fairness.fairness_metrics import (
    Independence,
    Separation,
    Sufficiency,
    DemographicParityWithoutGroundTruth,
)
