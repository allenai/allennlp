"""
A ``TokenIndexer`` determines how string tokens get represented as arrays of indices in a model.
"""

from allennlp.data.token_indexers.dep_label_indexer import DepLabelIndexer
from allennlp.data.token_indexers.ner_tag_indexer import NerTagIndexer
from allennlp.data.token_indexers.pos_tag_indexer import PosTagIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import OpenaiTransformerBytePairIndexer
from allennlp.data.token_indexers.wordpiece_indexer import WordpieceIndexer, PretrainedBertIndexer
from allennlp.data.token_indexers.spacy_indexer import SpacyTokenIndexer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
