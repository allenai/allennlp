"""
A `TokenEmbedder` is a `Module` that
embeds one-hot-encoded tokens as vectors.
"""

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders.empty_embedder import EmptyEmbedder
from allennlp.modules.token_embedders.bag_of_word_counts_token_embedder import (
    BagOfWordCountsTokenEmbedder,
)
from allennlp.modules.token_embedders.pass_through_token_embedder import PassThroughTokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import (
    PretrainedTransformerMismatchedEmbedder,
)
