import torch
from torch import allclose
import pytest
import math

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.fairness.bias_mitigators import (
    LinearBiasMitigator,
    HardBiasMitigator,
    INLPBiasMitigator,
    OSCaRBiasMitigator,
)


class LinearBiasMitigator(AllenNlpTestCase):
    def setup_method(self):
        he = [
            -0.03674268,
            -0.01102189,
            -0.11295283,
            -0.15441727,
            0.10571841,
            0.02682918,
            -0.15744929,
            0.1226158,
            -0.15828685,
            -0.03334491,
            0.02899621,
            0.08378106,
            -0.185853,
            -0.06560357,
            0.13508585,
            -0.0439771,
            -0.06198087,
            0.04707496,
            -0.14299142,
            0.01527495,
            0.03245981,
            0.16782729,
            0.11800925,
            -0.03638425,
            0.06842346,
            -0.50335569,
            -0.01674853,
            0.00737871,
            -0.01184865,
            -0.05754256,
            0.62074134,
            0.00821846,
            -0.10064919,
            -0.11947771,
            0.01908454,
            0.00299801,
            0.04459887,
            0.1844266,
            0.05744381,
            -0.06182177,
            -0.03095112,
            0.01870417,
            -0.11364226,
            0.03626173,
            -0.06610281,
            -0.04529561,
            -0.07130004,
            -0.06092753,
            -0.00761827,
            -0.00240861,
        ]
