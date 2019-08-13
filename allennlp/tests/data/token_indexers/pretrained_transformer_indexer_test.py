# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers import PretrainedTransformerIndexer


class TestPretrainedTransformerTokenizer):
    def test_as_array_produces_token_sequence(self):
        indexer = PretrainedTransformerIndexer("words")
        padded_tokens = indexer.as_padded_tensor({'key': [1, 2, 3, 4, 5]}, {'key': 10}, {})
        assert padded_tokens['key'].tolist() == [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
