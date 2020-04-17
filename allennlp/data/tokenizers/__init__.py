"""
This module contains various classes for performing
tokenization.
"""

from allennlp.data.tokenizers.tokenizer import Token, Tokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers.letters_digits_tokenizer import LettersDigitsTokenizer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.sentence_splitter import SentenceSplitter
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
