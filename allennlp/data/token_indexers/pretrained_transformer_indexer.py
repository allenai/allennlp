from typing import Dict, List, Tuple
import logging

from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

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
    num_added_start_tokens: ``int``, optional (default=1)
        The number of start tokens that the tokenizer adds to a sequence.
    num_added_end_tokens: ``int``, optional (default=1)
        The number of end tokens that the tokenizer adds to a sequence.
    intra_word_tokenization: ``bool``, optional (default=False)
        If True, further tokenize each token into subword tokens using the corresponding tokenizer.
        Should be set to the same value as the ``intra_word_tokenized`` option on the
        :class:`PretrainedTransformerEmbedder`.
    """

    def __init__(
        self,
        model_name: str,
        namespace: str = "tags",
        num_added_start_tokens: int = 1,
        num_added_end_tokens: int = 1,
        intra_word_tokenization: bool = False,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self._namespace = namespace
        # add_special_tokens=False for intra_word_tokenization
        self._tokenizer = PretrainedTransformerTokenizer(model_name, add_special_tokens=False)
        self._wrapped_tokenizer = self._tokenizer.tokenizer  # wrapped huggingface tokenizer
        self._num_added_start_tokens = num_added_start_tokens
        self._num_added_end_tokens = num_added_end_tokens
        assert (
            self._num_added_start_tokens + self._num_added_end_tokens
            == self._wrapped_tokenizer.num_added_tokens()
        )

        self._added_to_vocabulary = False

        self._intra_word_tokenization = intra_word_tokenization

    def _add_encoding_to_vocabulary(self, vocab: Vocabulary) -> None:
        """
        Copies tokens from ```transformers``` model to the specified namespace.
        Transformers vocab is taken from the <vocab>/<encoder> keys of the tokenizer object.
        """
        vocab_field_name = None
        if hasattr(self._wrapped_tokenizer, "vocab"):
            vocab_field_name = "vocab"
        elif hasattr(self._wrapped_tokenizer, "encoder"):
            vocab_field_name = "encoder"
        else:
            logger.warning(
                """Wasn't able to fetch vocabulary from pretrained transformers lib.
                Neither <vocab> nor <encoder> are the valid fields for vocab.
                Your tokens will still be correctly indexed, but vocabulary file will not be saved."""
            )
        if vocab_field_name is not None:
            pretrained_vocab = getattr(self._wrapped_tokenizer, vocab_field_name)
            for word, idx in pretrained_vocab.items():
                vocab._token_to_index[self._namespace][word] = idx
                vocab._index_to_token[self._namespace][idx] = word

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        # We need to do this before intra-word tokenization because we always want it to be
        # token-level masks. We create a separate wordpiece-level mask below. Note that in the case
        # of intra-word tokenization, wordpiece-lebel masks include special tokens while word-level
        # masks do not.
        mask = [1] * len(tokens)

        offsets = None
        if self._intra_word_tokenization:
            tokens, offsets = self._intra_word_tokenize(tokens)

        indices: List[int] = []
        for token in tokens:
            if getattr(token, "text_id", None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead. Id comes from the pretrained vocab.
                # It is computed in PretrainedTransformerTokenizer.
                indices.append(token.text_id)
            else:
                raise KeyError(
                    "Using PretrainedTransformerIndexer but field text_id is not set"
                    f" for the following token: {token.text}"
                )

        if self._intra_word_tokenization:
            # self._intra_word_tokenize() does not insert special tokens, so we need to do it here
            indices = self._wrapped_tokenizer.build_inputs_with_special_tokens(indices)
            offsets = [
                (start + self._num_added_start_tokens, end + self._num_added_start_tokens)
                for start, end in offsets
            ]

        output = {"token_ids": indices, "mask": mask}
        if self._intra_word_tokenization:
            output["wordpiece_mask"] = [1] * len(indices)
            output["offsets"] = offsets  # type: ignore
        return output

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        output: IndexedTokenList = {"token_ids": [], "mask": []}
        if self._intra_word_tokenization:
            output["wordpiece_mask"] = []
            output["offsets"] = []
        return output

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for key, val in tokens.items():
            tensor_dict[key] = torch.LongTensor(
                pad_sequence_to_length(
                    val,
                    padding_lengths[key],
                    default_value=lambda: (0, 0) if key == "offsets" else 0,
                )
            )
        return tensor_dict

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

    def _intra_word_tokenize(
        self, tokens: List[Token]
    ) -> Tuple[List[Token], List[Tuple[int, int]]]:
        """
        Tokenizes each word into wordpieces separately. Also calculates offsets such that
        wordpices[offsets[i][0]:offsets[i][1]] corresponds to the original i-th token.
        Does not insert special tokens.
        """
        wordpieces: List[Token] = []
        offsets = []
        cumulative = 0
        for token in tokens:
            subword_wordpieces = self._tokenizer.tokenize(token.text)
            wordpieces.extend(subword_wordpieces)

            start_offset = cumulative
            cumulative += len(subword_wordpieces)
            end_offset = cumulative - 1  # inclusive
            offsets.append((start_offset, end_offset))

        return wordpieces, offsets
