import logging
from typing import List, Tuple

from overrides import overrides
import pytorch_transformers
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@Tokenizer.register("wordpiece")
class WordpieceTokenizer(Tokenizer):
    """
    A ``WordpieceTokenizer`` splits a string of text (consistently of multiple words) into a list
    of wordpieces, where each word piece represents whether or not it is a continuation.  E.g.,
    ``'AllenNLP is awesome'`` might get split into ``['Allen', '##NL', '##P', 'is', 'awesome']``.

    Because tokenizing into wordpieces requires first having obtained a wordpiece vocabulary, we
    leverage ``pytorch_transformers`` to provide a ``PreTrainedTokenizer``.  We take a model name
    here, which we will pass to ``PreTrainedTokenizer.from_pretrained``.

    Parameters
    ----------
    model_name : ``str``
        The name of the pretrained wordpiece tokenizer to use.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.  We try
        to be a little bit smart about defaults here - e.g., if your model name contains ``bert``,
        we by default add ``[CLS]`` at the beginning and ``[SEP]`` at the end.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """
    def __init__(self,
                 model_name: str,
                 do_lowercase: bool = True,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        if model_name.endswith("-cased") and do_lowercase:
            logger.warning("Your wordpiece model appears to be cased, "
                           "but your indexer is lowercasing tokens.")
        elif model_name.endswith("-uncased") and not do_lowercase:
            logger.warning("Your wordpiece model appears to be uncased, "
                           "but your indexer is not lowercasing tokens.")
        self._tokenizer = PreTrainedTokenizer.from_pretrained(model_name,
                                                              do_lower_case=do_lowercase)
        default_start_tokens, default_end_tokens = _guess_start_and_end_token_defaults(model_name)
        self._start_tokens = start_tokens if start_tokens is not None else default_start_tokens
        self._end_tokens = end_tokens if end_tokens is not None else default_end_tokens

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        Splits ``text`` into a sequence of wordpiece tokens.
        """
        # TODO(mattg): track character offsets.  Might be too challenging to do it here, given that
        # pytorch-transformers is dealing with the whitespace...
        token_strings = self._start_tokens + self._tokenizer.tokenize(text) + self._end_tokens
        return [Token(t) for t in token_strings]

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]


def _guess_start_and_end_token_defaults(model_name: str) -> Tuple[List[str], List[str]]:
    if 'bert' in model_name:
        return (['[CLS]'], ['[SEP]'])
    else:
        return ([], [])
