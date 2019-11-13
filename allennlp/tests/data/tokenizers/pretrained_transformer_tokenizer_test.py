from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


class TestPretrainedTransformerTokenizer(AllenNlpTestCase):
    def test_splits_cased(self):
        sentence = "A, [MASK] AllenNLP sentence."
        expected_tokens = ["A", ",", "[MASK]", "Allen", "##NL", "##P", "sentence", "."]
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens

    def test_splits_uncased(self):
        sentence = "A, [MASK] AllenNLP sentence."
        expected_tokens = ["a", ",", "[MASK]", "allen", "##nl", "##p", "sentence", "."]
        tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
