import numpy
import torch
from allennlp.modules.token_embedders import PassThroughTokenEmbedder
from allennlp.common.testing import AllenNlpTestCase


class TestBagOfWordCountsTokenEmbedder(AllenNlpTestCase):
    def test_pass_through_embedder(self):
        embedder = PassThroughTokenEmbedder(3)
        tensor = torch.randn([4, 3])
        numpy.testing.assert_equal(tensor.numpy(), embedder(tensor).numpy())
        assert embedder.get_output_dim() == 3
