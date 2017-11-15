# pylint: disable=no-self-use,invalid-name
import pytest
import torch

from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import OneToManyTextFieldEmbedder
from allennlp.common.testing import AllenNlpTestCase


class TestOneToManyTextFieldEmbedder(AllenNlpTestCase):
    def setUp(self):
        super(TestOneToManyTextFieldEmbedder, self).setUp()
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("1")
        self.vocab.add_token_to_namespace("2")
        self.vocab.add_token_to_namespace("3")
        self.vocab.add_token_to_namespace("4")
        params = Params({
                "words1": [
                    {
                        "type": "embedding",
                        "embedding_dim": 2
                    },
                    {
                        "type": "embedding",
                        "embedding_dim": 2
                    }
                ],
                "words2": [{
                        "type": "embedding",
                        "embedding_dim": 5
                        }],
                "words3": [{
                        "type": "embedding",
                        "embedding_dim": 3
                        }]
                })
        self.token_embedder = OneToManyTextFieldEmbedder.from_params(self.vocab, params)
        self.inputs = {
                "words1": Variable(torch.LongTensor([[0, 2, 3, 5]])),
                "words2": Variable(torch.LongTensor([[1, 4, 3, 2]])),
                "words3": Variable(torch.LongTensor([[1, 5, 1, 2]]))
                }

    def test_get_output_dim_aggregates_dimension_from_each_embedding(self):
        assert self.token_embedder.get_output_dim() == 12

    def test_from_params_raises_without_lists_of_token_embedders(self):
        params = Params({"words1": {"type": "embedding", "embedding_dim": 2}})
        with pytest.raises(ConfigurationError):
            OneToManyTextFieldEmbedder.from_params(self.vocab, params)

    def test_forward_asserts_input_field_match(self):
        self.inputs['words4'] = self.inputs['words3']
        del self.inputs['words3']
        with pytest.raises(ConfigurationError):
            self.token_embedder(self.inputs)
        self.inputs['words3'] = self.inputs['words4']
        del self.inputs['words4']

    def test_forward_concats_resultant_embeddings(self):
        assert self.token_embedder(self.inputs).size() == (1, 4, 12)
