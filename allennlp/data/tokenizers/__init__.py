from collections import OrderedDict

from .tokenizer import Tokenizer
from .word_tokenizer import WordTokenizer
from .character_tokenizer import CharacterTokenizer

# The first item added here will be used as the default in some cases.
tokenizers = OrderedDict()  # pylint: disable=invalid-name
tokenizers['words'] = WordTokenizer
tokenizers['characters'] = CharacterTokenizer
