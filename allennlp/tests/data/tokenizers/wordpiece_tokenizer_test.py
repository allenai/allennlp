# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordpieceTokenizer

class TestWordpieceTokenizer(AllenNlpTestCase):
    def test_splits_into_wordpieces(self):
        tokenizer = WordpieceTokenizer('bert-base-cased')
        sentence = "A, small sentence."
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        expected_tokens = ["<S1>", "<S2>", "A", ",", " ", "s", "m", "a", "l", "l", " ", "s", "e",
                           "n", "t", "e", "n", "c", "e", ".", '</S2>', '</S1>']
        assert tokens == expected_tokens
