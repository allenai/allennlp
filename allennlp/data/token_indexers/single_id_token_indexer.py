from typing import Dict, List, Optional
import itertools

from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList


_DEFAULT_VALUE = "THIS IS A REALLY UNLIKELY VALUE THAT HAS TO BE A STRING"


@TokenIndexer.register("single_id")
class SingleIdTokenIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as single integers.

    Registered as a `TokenIndexer` with name "single_id".

    # Parameters

    namespace : `Optional[str]`, optional (default=`"tokens"`)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.  If you
        explicitly pass in `None` here, we will skip indexing and vocabulary lookups.  This means
        that the `feature_name` you use must correspond to an integer value (like `text_id`, for
        instance, which gets set by some tokenizers, such as when using byte encoding).
    lowercase_tokens : `bool`, optional (default=`False`)
        If `True`, we will call `token.lower()` before getting an index for the token from the
        vocabulary.
    start_tokens : `List[str]`, optional (default=`None`)
        These are prepended to the tokens provided to `tokens_to_indices`.
    end_tokens : `List[str]`, optional (default=`None`)
        These are appended to the tokens provided to `tokens_to_indices`.
    feature_name : `str`, optional (default=`"text"`)
        We will use the :class:`Token` attribute with this name as input.  This is potentially
        useful, e.g., for using NER tags instead of (or in addition to) surface forms as your inputs
        (passing `ent_type_` here would do that).  If you use a non-default value here, you almost
        certainly want to also change the `namespace` parameter, and you might want to give a
        `default_value`.
    default_value : `str`, optional
        When you want to use a non-default `feature_name`, you sometimes want to have a default
        value to go with it, e.g., in case you don't have an NER tag for a particular token, for
        some reason.  This value will get used if we don't find a value in `feature_name`.  If this
        is not given, we will crash if a token doesn't have a value for the given `feature_name`, so
        that you don't get weird, silent errors by default.
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        namespace: Optional[str] = "tokens",
        lowercase_tokens: bool = False,
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        feature_name: str = "text",
        default_value: str = _DEFAULT_VALUE,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]
        self._feature_name = feature_name
        self._default_value = default_value

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if self.namespace is not None:
            text = self._get_feature_value(token)
            if self.lowercase_tokens:
                text = text.lower()
            counter[self.namespace][text] += 1

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            text = self._get_feature_value(token)
            if self.namespace is None:
                # We could have a check here that `text` is an int; not sure it's worth it.
                indices.append(text)  # type: ignore
            else:
                if self.lowercase_tokens:
                    text = text.lower()
                indices.append(vocabulary.get_token_index(text, self.namespace))

        return {"tokens": indices}

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"tokens": []}

    def _get_feature_value(self, token: Token) -> str:
        text = getattr(token, self._feature_name)
        if text is None:
            if self._default_value is not _DEFAULT_VALUE:
                text = self._default_value
            else:
                raise ValueError(
                    f"{token} did not have attribute {self._feature_name}. If you "
                    "want to ignore this kind of error, give a default value in the "
                    "constructor of this indexer."
                )
        return text
