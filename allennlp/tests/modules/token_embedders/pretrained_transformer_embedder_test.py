import torch

from allennlp.common import Params
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.common.testing import AllenNlpTestCase


class TestPretrainedTransformerEmbedder(AllenNlpTestCase):
    def test_forward_runs_when_initialized_from_params(self):
        # This code just passes things off to ``transformers``, so we only have a very simple
        # test.
        params = Params({"model_name": "bert-base-uncased"})
        embedder = PretrainedTransformerEmbedder.from_params(params)
        token_ids = torch.randint(0, 100, (1, 4))
        mask = torch.randint(0, 2, (1, 4))
        output = embedder(token_ids=token_ids, attention_mask=mask)
        assert tuple(output.size()) == (1, 4, 768)
