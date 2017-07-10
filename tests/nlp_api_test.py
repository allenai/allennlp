# pylint: disable=no-self-use,invalid-name
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp import NlpApi


class TestNlpApi(AllenNlpTestCase):
    def test_get_token_embedder_uses_constructor_arguments_correctly(self):
        api = NlpApi(token_embedders={'default': 1})
        assert api.get_token_embedder('default') == 1
        assert api.get_token_embedder('not present') is None

    def test_get_new_token_embedder_uses_constructor_arguments_correctly(self):
        api = NlpApi(token_embedder_fn=lambda: 2)
        assert api.get_new_token_embedder() == 2

    def test_get_seq2seq_encoder_uses_constructor_arguments_correctly(self):
        api = NlpApi(seq2seq_encoders={'default': 1})
        assert api.get_seq2seq_encoder('default') == 1
        assert api.get_seq2seq_encoder('not present') is None

    def test_get_new_seq2seq_encoder_uses_constructor_arguments_correctly(self):
        api = NlpApi(seq2seq_encoder_fn=lambda: 2)
        assert api.get_new_seq2seq_encoder() == 2

    def test_get_seq2vec_encoder_uses_constructor_arguments_correctly(self):
        api = NlpApi(seq2vec_encoders={'default': 1})
        assert api.get_seq2vec_encoder('default') == 1
        assert api.get_seq2vec_encoder('not present') is None

    def test_get_new_seq2vec_encoder_uses_constructor_arguments_correctly(self):
        api = NlpApi(seq2vec_encoder_fn=lambda: 2)
        assert api.get_new_seq2vec_encoder() == 2
