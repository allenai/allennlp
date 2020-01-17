from typing import Dict, List
import logging

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
    token_min_padding_length: ``int``, optional (default=0)
        See superclass.
    """

    def __init__(
        self, model_name: str, namespace: str = "tags", token_min_padding_length: int = 0
    ) -> None:
        super().__init__(token_min_padding_length)
        self._namespace = namespace
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._added_to_vocabulary = False

        (
            self._num_added_start_tokens, self._num_added_end_tokens
        ) = self._determine_num_special_tokens_added()

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
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

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

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        mask = [1] * len(tokens)

        return {"token_ids": indices, "mask": mask}

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"token_ids": [], "mask": []}

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

    def _determine_num_special_tokens_added(self):
        """
        Determines the number of tokens self._tokenizer adds to a sequence (currently doesn't
        consider sequence pairs) in the start & end.
        """
        # Uses a slightly higher index to avoid tokenizer does special things to lower-indexed
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
