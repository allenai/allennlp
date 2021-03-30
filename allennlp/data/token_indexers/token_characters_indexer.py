from typing import Dict, List
import itertools
import warnings

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.tokenizers import Token, CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary


@TokenIndexer.register("characters")
class TokenCharactersIndexer(TokenIndexer):
    """
    This :class:`TokenIndexer` represents tokens as lists of character indices.

    Registered as a `TokenIndexer` with name "characters".

    # Parameters

    namespace : `str`, optional (default=`token_characters`)
        We will use this namespace in the :class:`Vocabulary` to map the characters in each token
        to indices.
    character_tokenizer : `CharacterTokenizer`, optional (default=`CharacterTokenizer()`)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        `CharacterTokenizer` with its default parameters, which uses unicode characters and
        retains casing.
    start_tokens : `List[str]`, optional (default=`None`)
        These are prepended to the tokens provided to `tokens_to_indices`.
    end_tokens : `List[str]`, optional (default=`None`)
        These are appended to the tokens provided to `tokens_to_indices`.
    min_padding_length : `int`, optional (default=`0`)
        We use this value as the minimum length of padding. Usually used with :class:`CnnEncoder`, its
        value should be set to the maximum value of `ngram_filter_sizes` correspondingly.
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        namespace: str = "token_characters",
        character_tokenizer: CharacterTokenizer = CharacterTokenizer(),
        start_tokens: List[str] = None,
        end_tokens: List[str] = None,
        min_padding_length: int = 0,
        token_min_padding_length: int = 0,
    ) -> None:
        super().__init__(token_min_padding_length)
        if min_padding_length == 0:
            url = "https://github.com/allenai/allennlp/issues/1954"
            warnings.warn(
                "You are using the default value (0) of `min_padding_length`, "
                f"which can cause some subtle bugs (more info see {url}). "
                "Strongly recommend to set a value, usually the maximum size "
                "of the convolutional layer size when using CnnEncoder.",
                UserWarning,
            )
        self._min_padding_length = min_padding_length
        self._namespace = namespace
        self._character_tokenizer = character_tokenizer

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError("TokenCharactersIndexer needs a tokenizer that retains text")
        for character in self._character_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            if getattr(character, "text_id", None) is None:
                counter[self._namespace][character.text] += 1

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[List[int]]]:
        indices: List[List[int]] = []
        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            token_indices: List[int] = []
            if token.text is None:
                raise ConfigurationError(
                    "TokenCharactersIndexer needs a tokenizer that retains text"
                )
            for character in self._character_tokenizer.tokenize(token.text):
                if getattr(character, "text_id", None) is not None:
                    # `text_id` being set on the token means that we aren't using the vocab, we just
                    # use this id instead.
                    index = character.text_id
                else:
                    index = vocabulary.get_token_index(character.text, self._namespace)
                token_indices.append(index)
            indices.append(token_indices)
        return {"token_characters": indices}

    @overrides
    def get_padding_lengths(self, indexed_tokens: IndexedTokenList) -> Dict[str, int]:
        padding_lengths = {}
        padding_lengths["token_characters"] = max(
            len(indexed_tokens["token_characters"]), self._token_min_padding_length
        )
        max_num_characters = self._min_padding_length
        for token in indexed_tokens["token_characters"]:
            max_num_characters = max(len(token), max_num_characters)  # type: ignore
        padding_lengths["num_token_characters"] = max_num_characters
        return padding_lengths

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        # Pad the tokens.
        padded_tokens = pad_sequence_to_length(
            tokens["token_characters"],
            padding_lengths["token_characters"],
            default_value=lambda: [],
        )

        # Pad the characters within the tokens.
        desired_token_length = padding_lengths["num_token_characters"]
        longest_token: List[int] = max(tokens["token_characters"], key=len, default=[])  # type: ignore
        padding_value = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" so we can take advantage of the fast implementation of itertools.zip_longest.
            padded_tokens.append([padding_value] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_value)))
        if desired_token_length > len(longest_token):
            # Removes the "dummy token".
            padded_tokens.pop()
        # Truncates all the tokens to the desired length, and return the result.
        return {
            "token_characters": torch.LongTensor(
                [list(token[:desired_token_length]) for token in padded_tokens]
            )
        }

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"token_characters": []}
