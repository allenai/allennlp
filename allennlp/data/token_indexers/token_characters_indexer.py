from typing import Dict, List
import itertools

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer


@TokenIndexer.register("characters")
class TokenCharactersIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents tokens as lists of character indices.

    Parameters
    ----------
    namespace : ``str``, optional (default=``token_characters``)
        We will use this namespace in the :class:`Vocabulary` to map the characters in each token
        to indices.
    character_tokenizer : ``CharacterTokenizer``, optional (default=``CharacterTokenizer()``)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
        retains casing.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'token_characters',
                 character_tokenizer: CharacterTokenizer = CharacterTokenizer()) -> None:
        self._namespace = namespace
        self._character_tokenizer = character_tokenizer

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('TokenCharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            if getattr(character, 'text_id', None) is None:
                counter[self._namespace][character.text] += 1

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> List[int]:
        indices = []
        if token.text is None:
            raise ConfigurationError('TokenCharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            if getattr(character, 'text_id', None) is not None:
                # `text_id` being set on the token means that we aren't using the vocab, we just
                # use this id instead.
                index = character.text_id
            else:
                index = vocabulary.get_token_index(character.text, self._namespace)
            indices.append(index)
        return indices

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        return {'num_token_characters': len(token)}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @overrides
    def pad_token_sequence(self,
                           tokens: List[List[int]],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[List[int]]:
        padded_tokens = pad_sequence_to_length(tokens, desired_num_tokens, default_value=lambda: [])
        desired_token_length = padding_lengths['num_token_characters']
        longest_token: List[int] = max(tokens, key=len, default=[])
        padding_index = 0
        if desired_token_length > len(longest_token):
            # Since we want to pad to greater than the longest token, we add a
            # "dummy token" to get the speed of itertools.zip_longest.
            padded_tokens.append([padding_index] * desired_token_length)
        # pad the list of lists to the longest sublist, appending 0's
        padded_tokens = list(zip(*itertools.zip_longest(*padded_tokens, fillvalue=padding_index)))
        if desired_token_length > len(longest_token):
            # now we remove the "dummy token" if we appended one.
            padded_tokens.pop()

        # Now we need to truncate all of them to our desired length, and return the result.
        return [list(token[:desired_token_length]) for token in padded_tokens]

    @classmethod
    def from_params(cls, params: Params) -> 'TokenCharactersIndexer':
        """
        Parameters
        ----------
        namespace : ``str``, optional (default=``token_characters``)
            We will use this namespace in the :class:`Vocabulary` to map the characters in each token
            to indices.
        character_tokenizer : ``Params``, optional (default=``Params({})``)
            We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
            options for byte encoding and other things.  These parameters get passed to the character
            tokenizer.  The default is to use unicode characters and to retain casing.
        """
        namespace = params.pop('namespace', 'token_characters')
        character_tokenizer_params = params.pop('character_tokenizer', {})
        character_tokenizer = CharacterTokenizer.from_params(character_tokenizer_params)
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace, character_tokenizer=character_tokenizer)
