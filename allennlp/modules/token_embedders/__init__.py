"""
A :class:`~allennlp.modules.token_embedders.token_embedder.TokenEmbedder` is a ``Module`` that
embeds one-hot-encoded tokens as vectors.
"""

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders.openai_transformer_embedder import OpenaiTransformerEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder
