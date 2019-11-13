import logging
from typing import List

from overrides import overrides
from transformers.tokenization_auto import AutoTokenizer

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@Tokenizer.register("pretrained_transformer")
class PretrainedTransformerTokenizer(Tokenizer):
    """
    A ``PretrainedTransformerTokenizer`` uses a model from HuggingFace's
    ``transformers`` library to tokenize some input text.  This often means wordpieces
    (where ``'AllenNLP is awesome'`` might get split into ``['Allen', '##NL', '##P', 'is',
    'awesome']``), but it could also use byte-pair encoding, or some other tokenization, depending
    on the pretrained model that you're using.

    We take a model name as an input parameter, which we will pass to
    ``AutoTokenizer.from_pretrained``.

    By default we add correct start and end tokens in the token indexer (depending on transformer model you
    have chosen). All you have to do is to use DEFAULT_SENTENCE_PAIR_SEPARATION_TOKEN to separate
    1st and 2nd sentence in your dataset reader.
    Note that the token has to be added after applying this tokenizer.

    Parameters
    ----------
    model_name : ``str``
        The name of the pretrained wordpiece tokenizer to use.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    """

    def __init__(
        self, model_name: str, start_tokens: List[str] = None, end_tokens: List[str] = None
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._start_tokens = start_tokens or []
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # TODO(mattg): track character offsets.  Might be too challenging to do it here, given that
        # ``transformers`` is dealing with the whitespace...
        token_strings = self._start_tokens + self._tokenizer.tokenize(text) + self._end_tokens
        return [Token(t) for t in token_strings]
