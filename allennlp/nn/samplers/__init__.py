"""
This module contains classes representing samplers which leverage the
multinomial distribution to sample point(s) given log-probabilities
"""

from allennlp.nn.samplers.sampler import Sampler
from allennlp.nn.samplers.samplers import MultinomialSampler
from allennlp.nn.samplers.samplers import TopKSampler
from allennlp.nn.samplers.samplers import TopPSampler
