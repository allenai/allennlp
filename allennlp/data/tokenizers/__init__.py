from collections import OrderedDict
from typing import Dict, Type  # pylint: disable=unused-import

from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer

# The first item added here will be used as the default in some cases.
# pylint: disable=invalid-name
tokenizers = OrderedDict()  # type: Dict[str, Type[Tokenizer]]
# pylint: enable=invalid-name

tokenizers['words'] = WordTokenizer
tokenizers['characters'] = CharacterTokenizer
