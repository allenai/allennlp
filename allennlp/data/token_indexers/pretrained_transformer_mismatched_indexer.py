from typing import Dict, List
import logging

from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer
from allennlp.data.token_indexers.token_indexer import IndexedTokenList

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

    Registered as a `TokenIndexer` with name "pretrained_transformer_mismatched".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use.
    namespace : `str`, optional (default=`tags`)
        We will add the tokens in the pytorch_transformer vocabulary to this vocabulary namespace.
        We use a somewhat confusing default value of `tags` so that we do not add padding or UNK
        tokens to this namespace, which would break on loading because we wouldn't find our default
        OOV token.
    max_length : `int`, optional (default = `None`)
        If positive, split the document into segments of this many tokens (including special tokens)
        before feeding into the embedder. The embedder embeds these segments independently and
        concatenate the results to get the original document representation. Should be set to
        the same value as the `max_length` option on the `PretrainedTransformerMismatchedEmbedder`.
    """

    def __init__(
        self, model_name: str, namespace: str = "tags", max_length: int = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # The matched version v.s. mismatched
        self._matched_indexer = PretrainedTransformerIndexer(
            model_name, namespace, max_length, **kwargs
        )
        self._allennlp_tokenizer = self._matched_indexer._allennlp_tokenizer
        self._tokenizer = self._matched_indexer._tokenizer
        self._num_added_start_tokens = self._matched_indexer._num_added_start_tokens
        self._num_added_end_tokens = self._matched_indexer._num_added_end_tokens

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        return self._matched_indexer.count_vocab_items(token, counter)

    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        self._matched_indexer._add_encoding_to_vocabulary_if_needed(vocabulary)

        wordpieces, offsets = self._allennlp_tokenizer.intra_word_tokenize([t.text for t in tokens])

        # For tokens that don't correspond to any word pieces, we put (-1, -1) into the offsets.
        # That results in the embedding for the token to be all zeros.
        offsets = [x if x is not None else (-1, -1) for x in offsets]

        output: IndexedTokenList = {
            "token_ids": [t.text_id for t in wordpieces],
            "mask": [True] * len(tokens),  # for original tokens (i.e. word-level)
            "type_ids": [t.type_id for t in wordpieces],
            "offsets": offsets,
            "wordpiece_mask": [True] * len(wordpieces),  # for wordpieces (i.e. subword-level)
        }

        return self._matched_indexer._postprocess_output(output)

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
                if key == "_tokenizer":
                    # This is a reference to a function in the huggingface code, which we can't
                    # really modify to make this clean.  So we special-case it.
                    continue
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            return True
        return NotImplemented
