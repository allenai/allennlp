from copy import deepcopy

import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.common.testing import AllenNlpTestCase


class TestTokenCharactersEncoder(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("1", "token_characters")
        self.vocab.add_token_to_namespace("2", "token_characters")
        self.vocab.add_token_to_namespace("3", "token_characters")
        self.vocab.add_token_to_namespace("4", "token_characters")
        params = Params(
            {
                "embedding": {"embedding_dim": 2, "vocab_namespace": "token_characters"},
                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 2,
                    "num_filters": 4,
                    "ngram_filter_sizes": [1, 2],
                    "output_dim": 3,
                },
            }
        )
        self.encoder = TokenCharactersEncoder.from_params(vocab=self.vocab, params=deepcopy(params))
        self.embedding = Embedding.from_params(vocab=self.vocab, params=params["embedding"])
        self.inner_encoder = Seq2VecEncoder.from_params(params["encoder"])
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 1.0}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(self.encoder)
        initializer(self.embedding)
        initializer(self.inner_encoder)

    def test_get_output_dim_uses_encoder_output_dim(self):
        assert self.encoder.get_output_dim() == 3

    def test_forward_applies_embedding_then_encoder(self):
        numpy_tensor = numpy.random.randint(6, size=(3, 4, 7))
        inputs = torch.from_numpy(numpy_tensor)
        encoder_output = self.encoder(inputs)
        reshaped_input = inputs.view(12, 7)
        embedded = self.embedding(reshaped_input)
        mask = (inputs != 0).long().view(12, 7)
        reshaped_manual_output = self.inner_encoder(embedded, mask)
        manual_output = reshaped_manual_output.view(3, 4, 3)
        assert_almost_equal(encoder_output.data.numpy(), manual_output.data.numpy())
