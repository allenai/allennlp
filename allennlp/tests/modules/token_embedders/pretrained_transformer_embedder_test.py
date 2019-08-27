# pylint: disable=no-self-use,invalid-name
import torch

from allennlp.common import Params
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.common.testing import AllenNlpTestCase

class TestPretrainedTransformerEmbedder(AllenNlpTestCase):
    def test_forward_runs_when_initialized_from_params(self):
        # This code just passes things off to pytorch-transformers, so we only have a very simple
        # test.
        params = Params({'model_name': 'bert-base-uncased'})
        embedder = PretrainedTransformerEmbedder.from_params(params)
        tensor = torch.randint(0, 100, (1, 4))
        output = embedder(tensor)
        assert tuple(output.size()) == (1, 4, 768)
