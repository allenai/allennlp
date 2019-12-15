"""
This module contains various classes for performing
tokenization.
"""

from allennlp.data.tokenizers.tokenizer import Token, Tokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers.letters_digits_tokenizer import LettersDigitsTokenizer
from allennlp.data.tokenizers.huggingface_transformers_tokenizer import (
    HuggingfaceTransformersTokenizer,
)
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.pretrained_transformer_pre_tokenizer import (
    OpenAIPreTokenizer,
    BertPreTokenizer,
)
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.sentence_splitter import SentenceSplitter
