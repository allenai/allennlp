from typing import Dict, List, Tuple, Union
import logging
import torch
from allennlp.common.util import pad_sequence_to_length

from overrides import overrides
from transformers.tokenization_auto import AutoTokenizer

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer")
class PretrainedTransformerIndexer(TokenIndexer):
    """
    This `TokenIndexer` assumes that Tokens already have their indexes in them (see `text_id` field).
    We still require `model_name` because we want to form allennlp vocabulary from pretrained one.
    This `Indexer` is only really appropriate to use if you've also used a
    corresponding :class:`PretrainedTransformerTokenizer` to tokenize your input.  Otherwise you'll
    have a mismatch between your tokens and your vocabulary, and you'll get a lot of UNK tokens.

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = -1)
        If positive, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerEmbedder`.
    """

    def __init__(
        self, model_name: str, namespace: str = "tags", max_length: int = -1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._namespace = namespace
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._added_to_vocabulary = False

        (
            self._num_added_start_tokens,
            self._num_added_end_tokens,
        ) = self.__class__.determine_num_special_tokens_added(self._tokenizer)

        self._max_length = max_length
        if self._max_length > 0:
            self._effective_max_length = (  # we need to take into account special tokens
                self._max_length - self._tokenizer.num_added_tokens()
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )

    def _add_encoding_to_vocabulary(self, vocab: Vocabulary) -> None:
        """
        Copies tokens from ```transformers``` model to the specified namespace.
        Transformers vocab is taken from the <vocab>/<encoder> keys of the tokenizer object.
        """
        vocab_field_name = None
        if hasattr(self._tokenizer, "vocab"):
            vocab_field_name = "vocab"
        elif hasattr(self._tokenizer, "encoder"):
            vocab_field_name = "encoder"
        else:
            logger.warning(
                """Wasn't able to fetch vocabulary from pretrained transformers lib.
                Neither <vocab> nor <encoder> are the valid fields for vocab.
                Your tokens will still be correctly indexed, but vocabulary file will not be saved."""
            )
        if vocab_field_name is not None:
            pretrained_vocab = getattr(self._tokenizer, vocab_field_name)
            for word, idx in pretrained_vocab.items():
                vocab._token_to_index[self._namespace][word] = idx
                vocab._index_to_token[self._namespace][idx] = word

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Union[Token, int]], vocabulary: Vocabulary
    ) -> IndexedTokenList:
        """
        `tokens` may already be indices, in which case the token -> index step is a no-op, but other
        logic still takes place, e.g. long sequence splitting.

        `tokens` should have special tokens already inserted.
        """
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        indices: List[int] = []
        type_ids: List[int] = []
        for token in tokens:
            if isinstance(token, int):
                indices.append(token)
            elif getattr(token, "text_id", None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead. Id comes from the pretrained vocab.
                # It is computed in PretrainedTransformerTokenizer.
                indices.append(token.text_id)
            else:
                raise KeyError(
                    "Using PretrainedTransformerIndexer but field text_id is not set"
                    f" for the following token: {token.text}"
                )

            if type_ids is not None and getattr(token, "type_id", None) is not None:
                type_ids.append(token.type_id)
            else:
                type_ids = None

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        mask = [1] * len(indices)

        result = {"token_ids": indices, "mask": mask}
        if type_ids is not None:
            result["type_ids"] = type_ids

        if self._max_length > 0:
            # We prepare long indices by converting them to (assuming max_length == 5)
            # [CLS] A B C [SEP] [CLS] D E F [SEP] ...
            # Embedder is responsible for folding this 1-d sequence to 2-d and feed to the
            # transformer model.
            # TODO(zhaofengw): we aren't respecting word boundaries when segmenting wordpieces.

            # Strips original special tokens
            indices = indices[self._num_added_start_tokens : -self._num_added_end_tokens]
            # Folds indices
            folded_indices = [
                indices[i : i + self._effective_max_length]
                for i in range(0, len(indices), self._effective_max_length)
            ]
            # Adds special tokens to each segment
            folded_indices = [
                self._tokenizer.build_inputs_with_special_tokens(segment)
                for segment in folded_indices
            ]
            # Flattens
            indices = [i for segment in folded_indices for i in segment]

            result["token_ids"] = indices
            result["segment_concat_mask"] = [1] * len(indices)

        return result

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        result: IndexedTokenList = {"token_ids": [], "mask": [], "type_ids": []}
        if self._max_length > 0:
            result["segment_concat_mask"] = []
        return result

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        # Different transformers use different padding values for tokens, but for mask and type id, the padding
        # value is always 0.
        return {
            key: torch.LongTensor(
                pad_sequence_to_length(
                    val,
                    padding_lengths[key],
                    default_value=lambda: 0
                    if key in {"mask", "type_ids"}
                    else self._tokenizer.pad_token_id,
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

    @classmethod
    def determine_num_special_tokens_added(cls, tokenizer) -> Tuple[int, int]:
        """
        Determines the number of tokens `tokenizer` adds to a sequence (currently doesn't
        consider sequence pairs) in the start & end.

        # Parameters

        tokenizer : `transformers.tokenization_utils.PretrainedTokenizer`, required.
            We want to determine the number of added tokens by this tokenizer.

        # Returns

        The number of tokens (`int`) that are inserted in the start & end of a sequence.
        """
        # Uses a slightly higher index to avoid tokenizer doing special things to lower-indexed
        # tokens which might be special.
        dummy = [1000]
        inserted = tokenizer.build_inputs_with_special_tokens(dummy)

        num_start = num_end = 0
        seen_dummy = False
        for idx in inserted:
            if idx == dummy[0]:
                if seen_dummy:  # seeing it twice
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy = True
                continue

            if not seen_dummy:
                num_start += 1
            else:
                num_end += 1

        assert num_start + num_end == tokenizer.num_added_tokens()
        return num_start, num_end
