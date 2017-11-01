import logging
from typing import Dict, List, Set
import numpy as np
from allennlp.common.checks import ConfigurationError
from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class FeatureIndexer(TokenIndexer):

    def __init__(self, feature_extractor, default_value) -> None:
        self.feature_extractor = feature_extractor
        self.default_value = default_value
        self.feature_dim = self.default_value.shape

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary = None):
        features = self.feature_extractor(token)
        if not isinstance(features, np.ndarray):
            raise ConfigurationError('Feature function for FeatureIndexer should return numpy array')
        if features.shape != self.feature_dim:
            raise ConfigurationError("feature function should return " \
            "vectors of dimensionality {}, but got {}".format(self.feature_dim, features.shape))
        return features

    @overrides
    def count_vocab_items(self, token, counter):
        return

    @overrides
    def get_padding_token(self):
        return []

    @classmethod
    def from_params(cls, params: Params):
        return

    @overrides
    def get_padding_lengths(self, token):
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens,
                           desired_num_tokens,
                           padding_lengths):
        return pad_sequence_to_length(tokens, desired_num_tokens,
                                      default_value=lambda: self.default_value)