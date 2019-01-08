# pylint: disable=no-self-use,invalid-name
from copy import deepcopy

import numpy as np
from numpy.testing import assert_almost_equal
import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.token_embedders import BagOfWordsTokenEmbedder
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.common.testing import AllenNlpTestCase


class TestBowTokenEmbedder(AllenNlpTestCase):
    def setUp(self):
        super(TestBowTokenEmbedder, self).setUp()
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("1")
        self.vocab.add_token_to_namespace("2")
        self.vocab.add_token_to_namespace("3")
        self.vocab.add_token_to_namespace("4")
        

    def test_forward_calculates_bow_properly(self):
        params = Params({})
        embedder = BagOfWordsTokenEmbedder.from_params(self.vocab, params=params)
        numpy_tensor = np.array([self.vocab.get_token_index(x) for x in ["1", "2", "3"]])
        inputs = torch.from_numpy(numpy_tensor).unsqueeze(1)
        embedder_output = embedder(inputs)
        numpy_tensor = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
        manual_output = torch.from_numpy(numpy_tensor)
        assert_almost_equal(embedder_output.data.numpy(), manual_output.data.numpy())
    
    def test_projects_properly(self):
        params = Params({"projection_dim": 50})
        embedder = BagOfWordsTokenEmbedder.from_params(self.vocab, params=params)
        numpy_tensor = np.array([self.vocab.get_token_index(x) for x in ["1", "2", "3"]])
        inputs = torch.from_numpy(numpy_tensor).unsqueeze(1)
        embedder_output = embedder(inputs)
        assert embedder_output.shape[1] == 50
