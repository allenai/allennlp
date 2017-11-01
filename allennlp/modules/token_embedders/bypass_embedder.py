import logging
from allennlp.common.params import Params
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@TokenEmbedder.register("bypass_encoder")
class BypassEmbedder(TokenEmbedder):
    """
    An embedding module that does not any processing of the input and just bypasses it forward. It is useful when
    it is needed to extract arbirary features and they cannot be represented as an integer.
    It can be used in a combination with :class:`~allennlp.data.token_indexers.FeatureIndexer`, which allows to extract
    feature vector, given any feature_extractor function.

    Parameters
    ----------
    dimensionality : int
        The size of each feature vector

    Returns
    -------
    A BypassEmbedder module.

    """
    def __init__(self, dimensionality):
        super(BypassEmbedder, self).__init__()
        self.dimensionality = dimensionality

    @overrides
    def forward(self, inputs):
        return inputs

    @overrides
    def get_output_dim(self) -> int:
        return self.dimensionality

