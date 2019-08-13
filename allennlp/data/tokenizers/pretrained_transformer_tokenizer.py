import logging
from typing import List, Tuple

from overrides import overrides
from pytorch_transformers.tokenization_auto import AutoTokenizer

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@Tokenizer.register("pretrained_transformer")
class PretrainedTransformerTokenizer(Tokenizer):
    """
    A ``PretrainedTransformerTokenizer`` uses a model from HuggingFace's
    ``pytorch_transformers`` library to tokenize some input text.  This often means wordpieces
    (where ``'AllenNLP is awesome'`` might get split into ``['Allen', '##NL', '##P', 'is',
    'awesome']``), but it could also use byte-pair encoding, or some other tokenization, depending
    on the pretrained model that you're using.

    We take a model name as an input parameter, which we will pass to
    ``AutoTokenizer.from_pretrained``.

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
                 do_lowercase: bool,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None) -> None:
        if model_name.endswith("-cased") and do_lowercase:
            logger.warning("Your pretrained model appears to be cased, "
                           "but your tokenizer is lowercasing tokens.")
        elif model_name.endswith("-uncased") and not do_lowercase:
            logger.warning("Your pretrained model appears to be uncased, "
                           "but your tokenizer is not lowercasing tokens.")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lowercase)
        default_start_tokens, default_end_tokens = _guess_start_and_end_token_defaults(model_name)
        self._start_tokens = start_tokens if start_tokens is not None else default_start_tokens
        self._end_tokens = end_tokens if end_tokens is not None else default_end_tokens

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # TODO(mattg): track character offsets.  Might be too challenging to do it here, given that
        # pytorch-transformers is dealing with the whitespace...
        token_strings = self._start_tokens + self._tokenizer.tokenize(text) + self._end_tokens
        return [Token(t) for t in token_strings]


def _guess_start_and_end_token_defaults(model_name: str) -> Tuple[List[str], List[str]]:
    if 'bert' in model_name:
        return (['[CLS]'], ['[SEP]'])
    else:
        return ([], [])
