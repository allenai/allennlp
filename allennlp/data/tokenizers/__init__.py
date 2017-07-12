from collections import OrderedDict
from typing import Dict  # pylint: disable=unused-import

from .tokenizer import Tokenizer
from .word_tokenizer import WordTokenizer
from .character_tokenizer import CharacterTokenizer

# The first item added here will be used as the default in some cases.
# pylint: disable=invalid-name
tokenizers = OrderedDict()  # type: Dict[str, type]
# pylint: enable=invalid-name

tokenizers['words'] = WordTokenizer
tokenizers['characters'] = CharacterTokenizer
