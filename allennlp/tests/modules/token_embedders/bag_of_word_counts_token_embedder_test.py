# pylint: disable=no-self-use,invalid-name
import numpy as np
from numpy.testing import assert_almost_equal
import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders import BagOfWordCountsTokenEmbedder
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError

class TestBagOfWordCountsTokenEmbedder(AllenNlpTestCase):
    def setUp(self):
        super(TestBagOfWordCountsTokenEmbedder, self).setUp()
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("1")
        self.vocab.add_token_to_namespace("2")
        self.vocab.add_token_to_namespace("3")
        self.vocab.add_token_to_namespace("4")
        self.non_padded_vocab = Vocabulary(non_padded_namespaces=['tokens'])

    def test_forward_calculates_bow_properly(self):
        params = Params({})
        embedder = BagOfWordCountsTokenEmbedder.from_params(self.vocab, params=params)
        numpy_tensor = np.array([[2, 0], [3, 0], [4, 4]])
        inputs = torch.from_numpy(numpy_tensor)
        embedder_output = embedder(inputs)
        numpy_tensor = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 2, 0]])
        manual_output = torch.from_numpy(numpy_tensor).float()
        assert_almost_equal(embedder_output.data.numpy(), manual_output.data.numpy())

    def test_zeros_out_unknown_tokens(self):
        params = Params({"ignore_oov": True})
        embedder = BagOfWordCountsTokenEmbedder.from_params(self.vocab, params=params)
        numpy_tensor = np.array([[1, 5], [2, 0], [4, 4]])
        inputs = torch.from_numpy(numpy_tensor)
        embedder_output = embedder(inputs)
        numpy_tensor = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 2, 0]])
        manual_output = torch.from_numpy(numpy_tensor).float()
        assert_almost_equal(embedder_output.data.numpy(), manual_output.data.numpy())

    def test_ignore_oov_should_fail_on_non_padded_vocab(self):
        params = Params({"ignore_oov": True})
        self.assertRaises(ConfigurationError,
                          BagOfWordCountsTokenEmbedder.from_params,
                          self.non_padded_vocab,
                          params)

    def test_projects_properly(self):
        params = Params({"projection_dim": 50})
        embedder = BagOfWordCountsTokenEmbedder.from_params(self.vocab, params=params)
        numpy_tensor = np.array([[1, 0], [1, 0], [4, 4]])
        inputs = torch.from_numpy(numpy_tensor)
        embedder_output = embedder(inputs)
        assert embedder_output.shape[1] == 50
