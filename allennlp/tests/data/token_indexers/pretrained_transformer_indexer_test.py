# pylint: disable=no-self-use,invalid-name
from pytorch_transformers.tokenization_auto import AutoTokenizer

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerIndexer


class TestPretrainedTransformerIndexer(AllenNlpTestCase):
    def test_as_array_produces_token_sequence(self):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lowercase=True)
        indexer = PretrainedTransformerIndexer(model_name='bert-base-uncased', do_lowercase=True)
        tokens = tokenizer.tokenize('AllenNLP is great')
        expected_ids = tokenizer.convert_tokens_to_ids(tokens)
        allennlp_tokens = [Token(token) for token in tokens]
        vocab = Vocabulary()
        indexed = indexer.tokens_to_indices(allennlp_tokens, vocab, 'key')
        assert indexed['key'] == expected_ids
