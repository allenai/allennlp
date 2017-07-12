from collections import OrderedDict
from typing import Dict, cast  # pylint: disable=unused-import

from .tokenizer import Tokenizer
from .word_tokenizer import WordTokenizer
from .character_tokenizer import CharacterTokenizer

# The first item added here will be used as the default in some cases.
# pylint: disable=invalid-name
tokenizers = OrderedDict()  # type: Dict[str, 'Tokenizer']
# pylint: enable=invalid-name

# these `cast`s are runtime no-ops that make `mypy` happy
tokenizers['words'] = cast(Tokenizer, WordTokenizer)
tokenizers['characters'] = cast(Tokenizer, CharacterTokenizer)
