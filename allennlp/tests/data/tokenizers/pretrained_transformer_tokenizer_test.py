# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import PretrainedTransformerTokenizer

class TestPretrainedTransformerTokenizer(AllenNlpTestCase):
    def test_splits_into_wordpieces(self):
        tokenizer = PretrainedTransformerTokenizer('bert-base-cased', do_lowercase=False)
        sentence = "A, [MASK] AllenNLP sentence."
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        expected_tokens = ["[CLS]", "A", ",", "[MASK]", "Allen", "##NL", "##P", "sentence", ".", "[SEP]"]
        assert tokens == expected_tokens
