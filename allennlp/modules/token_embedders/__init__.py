"""
A :class:`~allennlp.modules.token_embedders.token_embedder.TokenEmbedder` is a ``Module`` that
embeds one-hot-encoded tokens as vectors.
"""

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders.elmo_token_embedder_multilang import ElmoTokenEmbedderMultiLang
from allennlp.modules.token_embedders.openai_transformer_embedder import OpenaiTransformerEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder
from allennlp.modules.token_embedders.bidirectional_language_model_token_embedder import \
        BidirectionalLanguageModelTokenEmbedder
from allennlp.modules.token_embedders.language_model_token_embedder import \
        LanguageModelTokenEmbedder
from allennlp.modules.token_embedders.bag_of_word_counts_token_embedder import BagOfWordCountsTokenEmbedder
from allennlp.modules.token_embedders.pass_through_token_embedder import PassThroughTokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
