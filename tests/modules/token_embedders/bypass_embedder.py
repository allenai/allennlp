import torch
from torch.autograd import Variable
from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.token_embedders.bypass_embedder import BypassEmbedder
import numpy

class TestBypassEmbedding(AllenNlpTestCase):

    def test_bypass_works(self):
        embedder = BypassEmbedder(dimensionality = 1)
        input = numpy.array([[1.0],[1.0],[0.0],[0.0]], dtype=numpy.float32)
        input_tensor = Variable(torch.FloatTensor(input))
        embedded = embedder(input_tensor).data.numpy()
        assert numpy.allclose(embedded, input )