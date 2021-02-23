from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("whitespace")
@Tokenizer.register("just_spaces")
class WhitespaceTokenizer(Tokenizer):
    """
    A `Tokenizer` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.

    Note that we use `text.split()`, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.

    Registered as a `Tokenizer` with name "whitespace" and "just_spaces".
    """

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in text.split()]
