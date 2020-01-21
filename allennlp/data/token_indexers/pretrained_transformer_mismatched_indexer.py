from typing import Dict, List, Tuple
import logging

from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer
from allennlp.data.token_indexers.token_indexer import IndexedTokenList
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


@TokenIndexer.register("pretrained_transformer_mismatched")
class PretrainedTransformerMismatchedIndexer(TokenIndexer):
    """
    Use this indexer when (for whatever reason) you are not using a corresponding
    `PretrainedTransformerTokenizer` on your input. We assume that you used a tokenizer that splits
    strings into words, while the transformer expects wordpieces as input. This indexer splits the
    words into wordpieces and flattens them out. You should use the corresponding
    `PretrainedTransformerMismatchedEmbedder` to embed these wordpieces and then pull out a single
    vector for each original word.

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_len : `int`, optional (default = -1)
        If positive, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_len` option on the `PretrainedTransformerMismatchedEmbedder`.
    """

    def __init__(
        self, model_name: str, namespace: str = "tags", max_len: int = -1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # The matched version v.s. mismatched
        self._matched_indexer = PretrainedTransformerIndexer(
            model_name, namespace, max_len, **kwargs
        )
        self._num_added_start_tokens = self._matched_indexer._num_added_start_tokens
        self._num_added_end_tokens = self._matched_indexer._num_added_end_tokens
        self._tokenizer = self._matched_indexer._tokenizer

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        return self._matched_indexer.count_vocab_items(token, counter)

    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        orig_token_mask = [1] * len(tokens)
        tokens, offsets = self._intra_word_tokenize(tokens)
        wordpiece_mask = [1] * len(tokens)

        # {"token_ids": ..., "mask": ...}
        output = self._matched_indexer.tokens_to_indices(tokens, vocabulary)

        output["mask"] = orig_token_mask
        output["offsets"] = offsets
        output["wordpiece_mask"] = wordpiece_mask
        return output

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output = self._matched_indexer.get_empty_token_list()
        output["offsets"] = []
        output["wordpiece_mask"] = []
        return output

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tokens = tokens.copy()
        padding_lengths = padding_lengths.copy()

        offsets_tokens = tokens.pop("offsets")
        offsets_padding_lengths = padding_lengths.pop("offsets")

        tensor_dict = self._matched_indexer.as_padded_tensor_dict(tokens, padding_lengths)
        tensor_dict["offsets"] = torch.LongTensor(
            pad_sequence_to_length(
                offsets_tokens, offsets_padding_lengths, default_value=lambda: (0, 0)
            )
        )
        return tensor_dict

    def __eq__(self, other):
        if isinstance(other, PretrainedTransformerMismatchedIndexer):
            for key in self.__dict__:
                if key == "tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented

    def _intra_word_tokenize(
        self, tokens: List[Token]
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that wordpices[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        wordpieces: List[int] = []
        offsets = []
        cumulative = self._num_added_start_tokens
        for token in tokens:
            subword_wordpieces = self._tokenizer.encode(token.text, add_special_tokens=False)
            wordpieces.extend(subword_wordpieces)

            start_offset = cumulative
            cumulative += len(subword_wordpieces)
            end_offset = cumulative - 1  # inclusive
            offsets.append((start_offset, end_offset))

        wordpieces = self._tokenizer.build_inputs_with_special_tokens(wordpieces)
        assert cumulative + self._num_added_end_tokens == len(wordpieces)

        return wordpieces, offsets
