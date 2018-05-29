# pylint: disable=no-self-use
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer


class TestELMoTokenCharactersIndexer(AllenNlpTestCase):
    def test_bos_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.token_to_indices(Token('<S>'), Vocabulary())
        expected_indices = [259, 257, 260, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261]
        assert indices == expected_indices

    def test_eos_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.token_to_indices(Token('</S>'), Vocabulary())
        expected_indices = [259, 258, 260, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261]
        assert indices == expected_indices

    def test_unicode_to_char_ids(self):
        indexer = ELMoTokenCharactersIndexer()
        indices = indexer.token_to_indices(Token(chr(256) + 't'), Vocabulary())
        expected_indices = [259, 197, 129, 117, 260, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261, 261, 261, 261, 261,
                            261, 261, 261, 261, 261]
        assert indices == expected_indices

    def test_elmo_as_array_produces_token_sequence(self): # pylint: disable=invalid-name
        indexer = ELMoTokenCharactersIndexer()
        indices = [
                indexer.token_to_indices(Token(token), Vocabulary())
                for token in ['Second', '.']
        ]
        padded_tokens = indexer.pad_token_sequence(indices,
                                                   desired_num_tokens=3,
                                                   padding_lengths={})
        expected_padded_tokens = [[259, 84, 102, 100, 112, 111, 101, 260, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261],
                                  [259, 47, 260, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261, 261, 261, 261, 261,
                                   261, 261, 261, 261, 261],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0]]

        assert padded_tokens == expected_padded_tokens
