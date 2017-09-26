"""
This module contains classes representing regularization schemes
as well as a class for applying regularization to parameters.
"""

from allennlp.nn.regularizers.regularizer import Regularizer
from allennlp.nn.regularizers.regularizers import L1Regularizer
from allennlp.nn.regularizers.regularizers import L2Regularizer
from allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator
