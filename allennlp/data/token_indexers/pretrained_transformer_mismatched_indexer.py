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


@TokenIndexer.register("pretrained_transformer_pretokenized")
class PretrainedTransformerPretokenizedIndexer(PretrainedTransformerIndexer):
    """
    Use this indexer when input comes from pre-tokenized text and therefore
    `PretrainedTransformerTokenizer` was not used in the dataset loader. It tokenizes each token
    into wordpieces independently and concatenate them back together. Use it along with
    `PretrainedTransformerPretokenizedEmbedder`.
    """

    def __init__(
        self, model_name: str, namespace: str = "tags", token_min_padding_length: int = 0
    ) -> None:
        super().__init__(model_name, namespace, token_min_padding_length)
        # add_special_tokens=False sicne we don't want wordpieces to be surrounded by special tokens
        self._allennlp_tokenizer = PretrainedTransformerTokenizer(
            model_name, add_special_tokens=False
        )

        (
            self._num_added_start_tokens,
            self._num_added_end_tokens,
        ) = self._determine_num_special_tokens_added()

    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        orig_token_mask = [1] * len(tokens)
        tokens, offsets = self._intra_word_tokenize(tokens)

        # {"token_ids": ..., "mask": ...}
        output = super().tokens_to_indices(tokens, vocabulary)

        # self._intra_word_tokenize() does not insert special tokens, so we need to do it here
        output["token_ids"] = self._tokenizer.build_inputs_with_special_tokens(output["token_ids"])
        output["mask"] = orig_token_mask
        output["offsets"] = [
            (start + self._num_added_start_tokens, end + self._num_added_start_tokens)
            for start, end in offsets
        ]
        output["wordpiece_mask"] = [1] * len(output["token_ids"])
        return output

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output = super().get_empty_token_list()
        output["offsets"] = []
        output["wordpiece_mask"] = []
        return output

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        offsets_tokens = tokens.pop("offsets")
        offsets_padding_lengths = padding_lengths.pop("offsets")

        tensor_dict = super().as_padded_tensor_dict(tokens, padding_lengths)
        tensor_dict["offsets"] = torch.LongTensor(
            pad_sequence_to_length(
                offsets_tokens, offsets_padding_lengths, default_value=lambda: (0, 0)
            )
        )
        return tensor_dict

    def _intra_word_tokenize(
        self, tokens: List[Token]
    ) -> Tuple[List[Token], List[Tuple[int, int]]]:
        """
        Tokenizes each word into wordpieces separately. Also calculates offsets such that
        wordpices[offsets[i][0]:offsets[i][1] + 1] corresponds to the original i-th token.
        Does not insert special tokens.
        """
        wordpieces: List[Token] = []
        offsets = []
        cumulative = 0
        for token in tokens:
            subword_wordpieces = self._allennlp_tokenizer.tokenize(token.text)
            wordpieces.extend(subword_wordpieces)

            start_offset = cumulative
            cumulative += len(subword_wordpieces)
            end_offset = cumulative - 1  # inclusive
            offsets.append((start_offset, end_offset))

        return wordpieces, offsets

    def _determine_num_special_tokens_added(self):
        """
        Determines the number of tokens self._tokenizer adds to a sequence (currently doesn't
        consider sequence pairs) in the start & end.
        """
        # Uses a slightly higher index to avoid tokenizer doing special things to lower-indexed
        # tokens which might be special.
        dummy = [1000]
        inserted = self._tokenizer.build_inputs_with_special_tokens(dummy)

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

        assert num_start + num_end == self._tokenizer.num_added_tokens()
        return num_start, num_end
