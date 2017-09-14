# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import CharacterTokenizer

class TestCharacterTokenizer(AllenNlpTestCase):
    def test_splits_into_characters(self):
        tokenizer = CharacterTokenizer(start_tokens=['<S1>', '<S2>'], end_tokens=['</S2>', '</S1>'])
        sentence = "A, small sentence."
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        expected_tokens = ["<S1>", "<S2>", "A", ",", " ", "s", "m", "a", "l", "l", " ", "s", "e",
                           "n", "t", "e", "n", "c", "e", ".", '</S2>', '</S1>']
        assert tokens == expected_tokens

    def test_handles_byte_encoding(self):
        tokenizer = CharacterTokenizer(byte_encoding='utf-8', start_tokens=[259], end_tokens=[260])
        word = "åøâáabe"
        tokens = [t.text_id for t in tokenizer.tokenize(word)]
        # Note that we've added one to the utf-8 encoded bytes, to account for masking.
        expected_tokens = [259, 196, 166, 196, 185, 196, 163, 196, 162, 98, 99, 102, 260]
        assert tokens == expected_tokens
