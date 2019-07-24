from typing import List

from overrides import overrides
import pytorch_transformers
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("wordpiece")
class WordpieceTokenizer(Tokenizer):
    """
    A ``WordpieceTokenizer`` splits a string of text (consistently of multiple words) into a list
    of wordpieces, where each word piece represents whether or not it is a continuation.  E.g.,
    ``'AllenNLP is awesome'`` might get split into ``['Allen', '#N', '#L', '#P', 'is',
    'awesome']``.

    Because tokenizing into wordpieces requires first having obtained a wordpiece vocabulary, we
    leverage ``pytorch_transformers`` to provide a ``PreTrainedTokenizer``.  We take a model name
    here, which we will pass to ``PreTrainedTokenizer.from_pretrained``.

    Parameters
    ----------
    model_name : ``str``
        The name of the pretrained wordpiece tokenizer to use.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """
    def __init__(self,
                 model_name: str,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        print(PreTrainedTokenizer.max_model_input_sizes)
        self._tokenizer = PreTrainedTokenizer.from_pretrained(model_name)
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        Splits ``text`` into a sequence of wordpiece tokens.
        """
        return [Token(t) for t in self._tokenizer.tokenize(text)]

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]
