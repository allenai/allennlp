# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import pytest
import torch

from torch.autograd import Variable

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import ModelTextFieldEmbedder
from allennlp.testing.test_case import AllenNlpTestCase


class TestModelTextFieldEmbedder(AllenNlpTestCase):
    def setUp(self):
        super(TestModelTextFieldEmbedder, self).setUp()
        @Seq2SeqEncoder.register("no_op")
        class NoOp(Seq2SeqEncoder):
            def __init__(self, dim: int) -> None:
                super(NoOp, self).__init__()
                self._dim = dim

            def forward(self, inputs):
                return inputs

            def get_input_dim(self):
                return self._dim

            def get_output_dim(self):
                return self._dim

            @classmethod
            def from_params(cls, params: Params):
                dim = params.pop('dim')
                return cls(dim)

        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("1")
        self.vocab.add_token_to_namespace("2")
        self.vocab.add_token_to_namespace("3")
        self.vocab.add_token_to_namespace("4")
        self.vocab.add_token_to_namespace("B", namespace="tags")
        self.vocab.add_token_to_namespace("I", namespace="tags")
        self.vocab.add_token_to_namespace("O", namespace="tags")
        params = Params({
                "model": {
                        "type": "simple_tagger",
                        "text_field_embedder": {
                                "words1": {
                                        "type": "embedding",
                                        "embedding_dim": 2
                                        },
                                "words2": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                        },
                                "words3": {
                                        "type": "embedding",
                                        "embedding_dim": 3
                                        }
                                },
                        "stacked_encoder": {
                                "type": "no_op",
                                "dim": 10
                                }
                        },
                "input_name": "tokens",
                "output_name": "encoded_text",
                "output_dim": 10
                })
        self.text_field_embedder = ModelTextFieldEmbedder.from_params(self.vocab, params)
        self.inputs = {
                "words1": Variable(torch.LongTensor([[0, 2, 3, 5]])),
                "words2": Variable(torch.LongTensor([[1, 4, 3, 2]])),
                "words3": Variable(torch.LongTensor([[1, 5, 1, 2]]))
                }

    def tearDown(self):
        super(TestModelTextFieldEmbedder, self).tearDown()
        del Registrable._registry[Seq2SeqEncoder]["no_op"]  # pylint: disable=protected-access

    def test_get_output_dim_passes_through_correctly(self):
        assert self.text_field_embedder.get_output_dim() == 10

    def test_forward_passes_through_to_the_model(self):
        model = self.text_field_embedder._model  # pylint: disable=protected-access
        result = self.text_field_embedder(self.inputs).data.numpy()
        model_result = model.forward(tokens=self.inputs)['encoded_text'].data.numpy()
        assert result.shape == (1, 4, 10)
        assert_almost_equal(result, model_result)
