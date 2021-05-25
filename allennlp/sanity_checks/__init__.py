from allennlp.confidence_checks.verification_base import VerificationBase
from allennlp.confidence_checks.normalization_bias_verification import NormalizationBiasVerification

import warnings

warnings.warn(
    "Module 'sanity_checks' is deprecated, please use 'confidence_checks' instead.",
    DeprecationWarning,
)
