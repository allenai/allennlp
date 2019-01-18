# pylint: disable=no-self-use,invalid-name
import numpy as np
from numpy.testing import assert_almost_equal
import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders import BagOfWordCountsTokenEmbedder
from allennlp.common.testing import AllenNlpTestCase


class TestBagOfWordCountsTokenEmbedder(AllenNlpTestCase):
    def setUp(self):
        super(TestBagOfWordCountsTokenEmbedder, self).setUp()
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("1")
        self.vocab.add_token_to_namespace("2")
        self.vocab.add_token_to_namespace("3")
        self.vocab.add_token_to_namespace("4")

    def test_forward_calculates_bow_properly(self):
        params = Params({})
        embedder = BagOfWordCountsTokenEmbedder.from_params(self.vocab, params=params)
        numpy_tensor = np.array([[2, 0], [3, 0], [4, 4]])
        inputs = torch.from_numpy(numpy_tensor).unsqueeze(1)
        embedder_output = embedder(inputs)
        numpy_tensor = np.array([[1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 2, 0]])
        manual_output = torch.from_numpy(numpy_tensor).float()
        assert_almost_equal(embedder_output.data.numpy(), manual_output.data.numpy())

    def test_projects_properly(self):
        params = Params({"projection_dim": 50})
        embedder = BagOfWordCountsTokenEmbedder.from_params(self.vocab, params=params)
        numpy_tensor = np.array([self.vocab.get_token_index(x) for x in ["1", "2", "3"]])
        inputs = torch.from_numpy(numpy_tensor).unsqueeze(1)
        embedder_output = embedder(inputs)
        assert embedder_output.shape[1] == 50
