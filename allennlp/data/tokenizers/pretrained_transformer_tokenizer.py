import logging
from typing import List

from overrides import overrides
from transformers.tokenization_auto import AutoTokenizer

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

ALLENNLP_SENTENCE_PAIR_SEP = "@@SEP_SENTPAIR@@"


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

    We also add special tokens relative to the pretrained model and trunkate the sequences.

    This tokenizer also indexes tokens and adds the indexes as the ``Token`` fields so that
    they can be picked up by SingleIdTokenizer

    Parameters
    ----------
    model_name : ``str``
        The name of the pretrained wordpiece tokenizer to use.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    add_special_tokens: ``bool``, optional, (default=True)
        If set to ``True``, the sequences will be encoded with the special tokens relative
        to their model.
    max_length: ``int``, optional, (default=None)
        Ff set to a number, will limit the total sequence returned so that it has a maximum length.
        If there are overflowing tokens, those will be added to the returned dictionary
    stride: ``int``, optional, (default=0)
    If set to a number along with max_length, the overflowing tokens returned will contain some tokens
        from the main sequence returned. The value of this argument defines the number of additional tokens.
    truncation_strategy: ``str``, optional, (default='longest_first')
        String selected in the following options:
        - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
            starting from the longest one at each token (when there is a pair of input sequences)
        - 'only_first': Only truncate the first sequence
        - 'only_second': Only truncate the second sequence
        - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
    Argument descriptions are from
    https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691
    """

    def __init__(
        self,
        model_name: str,
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        add_special_tokens: bool = True,
        max_length: int = None,
        stride: int = 0,
        truncetion_strategy: str = "longest_first",
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._start_tokens = start_tokens
        self._end_tokens = end_tokens
        self._add_special_tokens = add_special_tokens
        self._max_length = max_length
        self._stride = stride
        self._truncation_strategy = truncetion_strategy

    def _add_custom_start_end_tokens(self, text: str) -> str:
        if self._start_tokens is not None:
            text = " ".join(self._start_tokens) + " " + text
        if self._end_tokens is not None:
            text = text + " " + " ".join(self._end_tokens)
        return text

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This can also handle sentence pairs. In this case 1st and 2nd sentences in ``text``
        are expected to be separated with ``ALLENNLP_SENTENCE_PAIR_SEP`` symbol.
        """
        # TODO(mattg): track character offsets.  Might be too challenging to do it here, given that
        # ``transformers``` is dealing with the whitespace...
        if ALLENNLP_SENTENCE_PAIR_SEP in text:
            sentence_1, sentence_2 = text.split(ALLENNLP_SENTENCE_PAIR_SEP)
            sentence_1 = self._add_custom_start_end_tokens(sentence_1)
            sentence_2 = self._add_custom_start_end_tokens(sentence_2)
        else:
            sentence_1 = text
            sentence_1 = self._add_custom_start_end_tokens(sentence_1)
            sentence_2 = None

        encoded_tokens = self._tokenizer.encode_plus(
            text=sentence_1,
            text_pair=sentence_2,
            add_special_tokens=self._add_special_tokens,
            max_length=self._max_length,
            stride=self._stride,
            truncation_strategy=self._truncation_strategy,
            return_tensors=None,
        )
        # token_ids containes final list with ids for both regualr and special tokens
        token_ids = encoded_tokens["input_ids"]
        token_type_ids = encoded_tokens["token_type_ids"]

        token_strings = self._tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

        tokens = []
        for token_str, token_id, token_type_id in zip(token_strings, token_ids, token_type_ids):
            tokens.append(Token(text=token_str, text_id=token_id, type_id=token_type_id))
        return tokens
