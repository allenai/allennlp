from typing import List

from overrides import overrides

from allennlp.common import Params
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

@Tokenizer.register("basic_lm")
class BasicLMTokenizer(Tokenizer):
    """
    A ``BasicLMTokenizer`` splits strings of text into tokens by treating space characters as 
    separator, and pad a special EOF token to the end of each sentence. This tokenizer is 
    relative fast and only provides the very basic function as a tokenizer (suitable for
    large, already tokenized corpora, e.g., the wiki103 dataset).
    """

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        if text and not text.isspace():
            return [Token(word) for word in text.split()] + [Token("<eof>")]
        else:
            return []

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [[Token(word) for word in words.split()] + [Token("<eof>")]
         if words and not words.isspace() else [] for words in texts]

    @classmethod
    def from_params(cls, params: Params) -> 'BasicLMTokenizer':
        params.assert_empty(cls.__name__)
        return cls()