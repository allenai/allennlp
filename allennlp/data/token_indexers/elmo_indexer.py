from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary


def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.

    We allow to add optional additional special tokens with designated
    character ids with ``tokens_to_add``.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260 # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
            beginning_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )
    end_of_sentence_characters = _make_bos_eos(
            end_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )

    bos_token = '<S>'
    eos_token = '</S>'

    def __init__(self, tokens_to_add: Dict[str, int] = None) -> None:
        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str) -> List[int]:
        if word in self.tokens_to_add:
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = ELMoCharacterMapper.end_of_word_character
        elif word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode('utf-8', 'ignore')[:(ELMoCharacterMapper.max_word_length-2)]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


@TokenIndexer.register("elmo_characters")
class ELMoTokenCharactersIndexer(TokenIndexer[List[int]]):
    """
    Convert a token to an array of character ids to compute ELMo representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    tokens_to_add : ``Dict[str, int]``, optional (default=``None``)
        If not None, then provides a mapping of special tokens to character
        ids. When using pre-trained models, then the character id must be
        less then 261, and we recommend using un-used ids (e.g. 1-32).
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'elmo_characters',
                 tokens_to_add: Dict[str, int] = None,
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self._namespace = namespace
        self._mapper = ELMoCharacterMapper(tokens_to_add)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        # TODO(brendanr): Retain the token to index mappings in the vocabulary and remove this
        # pylint pragma. See:
        # https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/wordpiece_indexer.py#L113

        # pylint: disable=unused-argument
        texts = [token.text for token in tokens]

        if any(text is None for text in texts):
            raise ConfigurationError('ELMoTokenCharactersIndexer needs a tokenizer '
                                     'that retains text')
        return {index_name: [self._mapper.convert_word_to_char_ids(text) for text in texts]}

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        # pylint: disable=unused-argument
        return {}

    @staticmethod
    def _default_value_for_padding():
        return [0] * ELMoCharacterMapper.max_word_length

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[List[int]]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        # pylint: disable=unused-argument
        return {key: torch.LongTensor(pad_sequence_to_length(
                val, desired_num_tokens[key], default_value=self._default_value_for_padding))
                for key, val in tokens.items()}
