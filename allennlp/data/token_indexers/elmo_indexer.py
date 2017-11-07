from typing import Dict, List

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary


def _make_bos_eos(char, pad_char, bow_char, eow_char, max_word_length):
    ret = [pad_char] * max_word_length
    ret[0] = bow_char
    ret[1] = char
    ret[2] = eow_char
    return ret

class ELMoCharacterMapper(object):
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    bos_char = 256  # <begin sentence>
    eos_char = 257  # <end sentence>
    bow_char = 258  # <begin word>
    eow_char = 259  # <end word>
    pad_char = 260 # <padding>

    bos_chars = _make_bos_eos(bos_char, pad_char, bow_char, eow_char, max_word_length)
    eos_chars = _make_bos_eos(eos_char, pad_char, bow_char, eow_char, max_word_length)

    bos_token = '<S>'
    eos_token = '</S>'

    @staticmethod
    def convert_word_to_char_ids(word):
        if word == ELMoCharacterMapper.bos_token:
            ret = ELMoCharacterMapper.bos_chars
        elif word == ELMoCharacterMapper.eos_token:
            ret = ELMoCharacterMapper.eos_chars
        else:
            word_encoded = word.encode('utf-8', 'ignore')[:(ELMoCharacterMapper.max_word_length-2)]
            ret = [ELMoCharacterMapper.pad_char] * ELMoCharacterMapper.max_word_length
            ret[0] = ELMoCharacterMapper.bow_char
            for k, chr_id in enumerate(word_encoded, start=1):
                ret[k] = chr_id
            ret[len(word_encoded) + 1] = ELMoCharacterMapper.eow_char

        # +1 one for masking
        return [c + 1 for c in ret]


@TokenIndexer.register("elmo_characters")
class ELMoTokenCharactersIndexer(TokenIndexer[List[int]]):
    """
    Convert a token to an array of character ids to compute ELMo representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'elmo_characters') -> None:
        self._namespace = namespace

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def token_to_indices(self, token: Token, vocabulary: Vocabulary) -> List[int]:
        #pylint: disable=unused-argument
        if token.text is None:
            raise ConfigurationError('ELMoTokenCharactersIndexer needs a tokenizer '
                                     'that retains text')
        return ELMoCharacterMapper.convert_word_to_char_ids(token.text)

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        #pylint: disable=unused-argument
        return {}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @staticmethod
    def _default_value_for_padding():
        return [0] * ELMoCharacterMapper.max_word_length

    @overrides
    def pad_token_sequence(self,
                           tokens: List[List[int]],
                           desired_num_tokens: int,
                           padding_lengths: Dict[str, int]) -> List[List[int]]:
        #pylint: disable=unused-argument
        return pad_sequence_to_length(tokens, desired_num_tokens,
                                      default_value=self._default_value_for_padding)

    @classmethod
    def from_params(cls, params: Params) -> 'ELMoTokenCharactersIndexer':
        """
        Parameters
        ----------
        namespace : ``str``, optional (default=``elmo_characters``)
        """
        namespace = params.pop('namespace', 'elmo_characters')
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace)
