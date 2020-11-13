from typing import Dict, List, Optional, Tuple, Any
import logging
import torch
from allennlp.common.util import pad_sequence_to_length

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
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

    Registered as a `TokenIndexer` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If not None, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerEmbedder`.
    tokenizer_kwargs : `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        namespace: str = "tags",
        max_length: int = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._namespace = namespace
        self._allennlp_tokenizer = PretrainedTransformerTokenizer(
            model_name, tokenizer_kwargs=tokenizer_kwargs
        )
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._added_to_vocabulary = False

        self._num_added_start_tokens = len(self._allennlp_tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(self._allennlp_tokenizer.single_sequence_end_tokens)

        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(self._allennlp_tokenizer.tokenize("a")) - 1
            self._effective_max_length = (  # we need to take into account special tokens
                self._max_length - num_added_tokens
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )

    def _add_encoding_to_vocabulary_if_needed(self, vocab: Vocabulary) -> None:
        """
        Copies tokens from ```transformers``` model's vocab to the specified namespace.
        """
        if self._added_to_vocabulary:
            return

        try:
            vocab_items = self._tokenizer.get_vocab().items()
        except NotImplementedError:
            vocab_items = (
                (self._tokenizer.convert_ids_to_tokens(idx), idx)
                for idx in range(self._tokenizer.vocab_size)
            )
        for word, idx in vocab_items:
            vocab._token_to_index[self._namespace][word] = idx
            vocab._index_to_token[self._namespace][idx] = word

        self._added_to_vocabulary = True

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        indices, type_ids = self._extract_token_and_type_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        output: IndexedTokenList = {
            "token_ids": indices,
            "mask": [True] * len(indices),
            "type_ids": type_ids or [0] * len(indices),
        }

        return self._postprocess_output(output)

    @overrides
    def indices_to_tokens(
        self, indexed_tokens: IndexedTokenList, vocabulary: Vocabulary
    ) -> List[Token]:
        self._add_encoding_to_vocabulary_if_needed(vocabulary)

        token_ids = indexed_tokens["token_ids"]
        type_ids = indexed_tokens.get("type_ids")

        return [
            Token(
                text=vocabulary.get_token_from_index(token_ids[i], self._namespace),
                text_id=token_ids[i],
                type_id=type_ids[i] if type_ids is not None else None,
            )
            for i in range(len(token_ids))
        ]

    def _extract_token_and_type_ids(self, tokens: List[Token]) -> Tuple[List[int], List[int]]:
        """
        Roughly equivalent to `zip(*[(token.text_id, token.type_id) for token in tokens])`,
        with some checks.
        """
        indices: List[int] = []
        type_ids: List[int] = []
        for token in tokens:
            indices.append(
                token.text_id
                if token.text_id is not None
                else self._tokenizer.convert_tokens_to_ids(token.text)
            )
            type_ids.append(token.type_id if token.type_id is not None else 0)
        return indices, type_ids

    def _postprocess_output(self, output: IndexedTokenList) -> IndexedTokenList:
        """
        Takes an IndexedTokenList about to be returned by `tokens_to_indices()` and adds any
        necessary postprocessing, e.g. long sequence splitting.

        The input should have a `"token_ids"` key corresponding to the token indices. They should
        have special tokens already inserted.
        """
        if self._max_length is not None:
            # We prepare long indices by converting them to (assuming max_length == 5)
            # [CLS] A B C [SEP] [CLS] D E F [SEP] ...
            # Embedder is responsible for folding this 1-d sequence to 2-d and feed to the
            # transformer model.
            # TODO(zhaofengw): we aren't respecting word boundaries when segmenting wordpieces.

            indices = output["token_ids"]
            # Strips original special tokens
            indices = indices[
                self._num_added_start_tokens : len(indices) - self._num_added_end_tokens
            ]
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

            output["token_ids"] = indices
            output["type_ids"] = [0] * len(indices)
            output["segment_concat_mask"] = [True] * len(indices)

        return output

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output: IndexedTokenList = {"token_ids": [], "mask": [], "type_ids": []}
        if self._max_length is not None:
            output["segment_concat_mask"] = []
        return output

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for key, val in tokens.items():
            if key == "type_ids":
                padding_value = 0
                mktensor = torch.LongTensor
            elif key == "mask" or key == "wordpiece_mask":
                padding_value = False
                mktensor = torch.BoolTensor
            elif len(val) > 0 and isinstance(val[0], bool):
                padding_value = False
                mktensor = torch.BoolTensor
            else:
                padding_value = self._tokenizer.pad_token_id
                if padding_value is None:
                    padding_value = (
                        0  # Some tokenizers don't have padding tokens and rely on the mask only.
                    )
                mktensor = torch.LongTensor

            tensor = mktensor(
                pad_sequence_to_length(
                    val, padding_lengths[key], default_value=lambda: padding_value
                )
            )

            tensor_dict[key] = tensor
        return tensor_dict

    def __eq__(self, other):
        if isinstance(other, PretrainedTransformerIndexer):
            for key in self.__dict__:
                if key == "_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
