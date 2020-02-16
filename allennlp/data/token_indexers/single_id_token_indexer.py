from typing import Dict, List
import itertools

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList


@TokenIndexer.register("single_id")
class SingleIdTokenIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    # Parameters

    namespace : `str`, optional (default=`tokens`)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : `bool`, optional (default=`False`)
        If `True`, we will call `token.lower()` before getting an index for the token from the
        vocabulary.
    start_tokens : `List[str]`, optional (default=`None`)
        These are prepended to the tokens provided to `tokens_to_indices`.
    end_tokens : `List[str]`, optional (default=`None`)
        These are appended to the tokens provided to `tokens_to_indices`.
    feature_name : `str`, optional (default=`text`)
        We will use the :class:`Token` attribute with this name as input
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        namespace: str = "tokens",
        lowercase_tokens: bool = False,
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        feature_name: str = "text",
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]
        self.feature_name = feature_name

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If `text_id` is set on the token (e.g., if we're using some kind of hash-based word
        # encoding), we will not be using the vocab for this token.
        if getattr(token, "text_id", None) is None:
            text = getattr(token, self.feature_name)
            if self.lowercase_tokens:
                text = text.lower()
            counter[self.namespace][text] += 1

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            if getattr(token, "text_id", None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just use
                # this id instead.
                indices.append(token.text_id)
            else:
                text = getattr(token, self.feature_name)
                if self.lowercase_tokens:
                    text = text.lower()
                indices.append(vocabulary.get_token_index(text, self.namespace))

        return {"tokens": indices}

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}
