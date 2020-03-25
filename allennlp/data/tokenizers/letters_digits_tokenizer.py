import re
from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("letters_digits")
class LettersDigitsTokenizer(Tokenizer):
    """
    A `Tokenizer` which keeps runs of (unicode) letters and runs of digits together, while
    every other non-whitespace character becomes a separate word.

    Registered as a `Tokenizer` with name "letters_digits".
    """

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # We use the [^\W\d_] pattern as a trick to match unicode letters
        tokens = [Token(m.group(), idx=m.start()) for m in re.finditer(r"[^\W\d_]+|\d+|\S", text)]
        return tokens
