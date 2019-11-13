from typing import Dict, List
import logging

from overrides import overrides
from transformers.tokenization_auto import AutoTokenizer
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import DEFAULT_SENTENCE_PAIR_SEPARATION_TOKEN

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer")
class PretrainedTransformerIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` uses a tokenizer from the ``transformers`` repository to
    index tokens.  This ``Indexer`` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    Parameters
    ----------
    model_name : ``str``
        The name of the ``transformers`` model to use.
    namespace : ``str``, optional (default=``tags``)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of ``tags`` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.

    See huggingface's ``encode`` function from tokenization_utils.py
    https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691
    for other arguments descriptions.
    """

    def __init__(
        self,
        model_name: str,
        namespace: str = "tags",
        token_min_padding_length: int = 0,
        add_special_tokens: bool = True,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
    ) -> None:
        super().__init__(token_min_padding_length)
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._namespace = namespace
        self._added_to_vocabulary = False
        self._padding_value = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        logger.info(f"Using token indexer padding value of {self._padding_value}")
        self._add_special_tokens = add_special_tokens
        self._max_length = max_length
        self._stride = stride
        self._truncation_strategy = truncation_strategy

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # TODO: figure out if the special tokens are already added to the vocabulary of pretrained model
        for word, idx in self.tokenizer.vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary and hasattr(self.tokenizer, "vocab"):
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True
        token_text = [token.text for token in tokens]
        if DEFAULT_SENTENCE_PAIR_SEPARATION_TOKEN in token_text:
            sep_pos = token_text.index(DEFAULT_SENTENCE_PAIR_SEPARATION_TOKEN)
            first_sentence = token_text[:sep_pos]
            second_sentence = token_text[sep_pos + 1 :]
        else:
            first_sentence = token_text
            second_sentence = None

        # Our input can be a single sentences or a pair of sentences.
        # In both cases, the output is always a single list of indexes.
        indices = self.tokenizer.encode(
            text=first_sentence,
            text_pair=second_sentence,
            add_special_tokens=self._add_special_tokens,
            max_length=self._max_length,
            stride=self._stride,
            truncation_strategy=self._truncation_strategy,
            return_tensors=False,
        )

        return {index_name: indices}

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:
        return {}

    @overrides
    def as_padded_tensor(
        self,
        tokens: Dict[str, List[int]],
        desired_num_tokens: Dict[str, int],
        padding_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        return {
            key: torch.LongTensor(
                pad_sequence_to_length(
                    val, desired_num_tokens[key], default_value=lambda: self._padding_value
                )
            )
            for key, val in tokens.items()
        }

    def __eq__(self, other):
        if isinstance(other, PretrainedTransformerIndexer):
            for key in self.__dict__:
                if key == "tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
