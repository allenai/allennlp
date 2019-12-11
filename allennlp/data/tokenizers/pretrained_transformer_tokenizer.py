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

    We also add special tokens relative to the pretrained model and truncate the sequences.

    This tokenizer also indexes tokens and adds the indexes to the ``Token`` fields so that
    they can be picked up by ``PretrainedTransformerIndexer``.

    Parameters
    ----------
    model_name : ``str``
        The name of the pretrained wordpiece tokenizer to use.
    add_special_tokens: ``bool``, optional, (default=True)
        If set to ``True``, the sequences will be encoded with the special tokens relative
        to their model.
    max_length: ``int``, optional (default=None)
        If set to a number, will limit the total sequence returned so that it has a maximum length.
        If there are overflowing tokens, those will be added to the returned dictionary
    stride: ``int``, optional (default=0)
        If set to a number along with max_length, the overflowing tokens returned will contain some tokens
        from the main sequence returned. The value of this argument defines the number of additional tokens.
    truncation_strategy: ``str``, optional (default='longest_first')
        String selected in the following options:
        - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
        starting from the longest one at each token (when there is a pair of input sequences)
        - 'only_first': Only truncate the first sequence
        - 'only_second': Only truncate the second sequence
        - 'do_not_truncate': Do not truncate (raise an error if the input sequence is longer than max_length)

    Argument descriptions are from
    https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691
    """

    def __init__(
        self,
        model_name: str,
        add_special_tokens: bool = True,
        max_length: int = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._add_special_tokens = add_special_tokens
        self._max_length = max_length
        self._stride = stride
        self._truncation_strategy = truncation_strategy

    def _tokenize(self, sentence_1: str, sentence_2: str = None):
        """
        This method works on both sentence and sentence pair.
        """
        # TODO(mattg): track character offsets.  Might be too challenging to do it here, given that
        # ``transformers``` is dealing with the whitespace...

        encoded_tokens = self._tokenizer.encode_plus(
            text=sentence_1,
            text_pair=sentence_2,
            add_special_tokens=self._add_special_tokens,
            max_length=self._max_length,
            stride=self._stride,
            truncation_strategy=self._truncation_strategy,
            return_tensors=None,
        )
        # token_ids contains a final list with ids for both regular and special tokens
        token_ids, token_type_ids = encoded_tokens["input_ids"], encoded_tokens["token_type_ids"]

        tokens = []
        for token_id, token_type_id in zip(token_ids, token_type_ids):
            token_str = self._tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False)
            tokens.append(Token(text=token_str, text_id=token_id, type_id=token_type_id))

        return tokens

    def tokenize_sentence_pair(self, sentence_1: str, sentence_2: str) -> List[Token]:
        """
        This methods properly handels a pair of sentences.
        """
        return self._tokenize(sentence_1, sentence_2)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This method only handels a single sentence (or sequence) of text.
        Refer to the ``tokenize_sentence_pair`` method if you have a sentence pair.
        """
        return self._tokenize(text)
