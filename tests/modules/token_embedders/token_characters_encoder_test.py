# pylint: disable=no-self-use,invalid-name
from copy import deepcopy

import numpy
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.training.initializers import InitializerApplicator
from allennlp.testing.test_case import AllenNlpTestCase


class TestTokenCharactersEncoder(AllenNlpTestCase):
    def setUp(self):
        super(TestTokenCharactersEncoder, self).setUp()
        self.vocab = Vocabulary()
        self.vocab.add_token_to_namespace("1", "token_characters")
        self.vocab.add_token_to_namespace("2", "token_characters")
        self.vocab.add_token_to_namespace("3", "token_characters")
        self.vocab.add_token_to_namespace("4", "token_characters")
        params = Params({
                "embedding": {
                        "embedding_dim": 2
                        },
                "encoder": {
                        "type": "lstm",
                        "input_size": 2,
                        "hidden_size": 3,
                        "num_layers": 1
                        }
                })
        self.encoder = TokenCharactersEncoder.from_params(self.vocab, deepcopy(params))
        self.embedding = Embedding.from_params(self.vocab, params["embedding"])
        self.inner_encoder = Seq2VecEncoder.from_params(params["encoder"])
        const_init = lambda tensor: torch.nn.init.constant(tensor, 1.)
        initializer = InitializerApplicator(default_initializer=const_init)
        initializer(self.encoder)
        initializer(self.embedding)
        initializer(self.inner_encoder)

    def test_get_output_dim_uses_encoder_output_dim(self):
        assert self.encoder.get_output_dim() == 3

    def test_forward_applies_embedding_then_encoder(self):
        numpy_tensor = numpy.random.randint(6, size=(3, 4, 7))
        inputs = Variable(torch.from_numpy(numpy_tensor))
        encoder_output = self.encoder(inputs)
        reshaped_input = inputs.view(12, 7)
        reshaped_manual_output = self.inner_encoder(self.embedding(reshaped_input))
        manual_output = reshaped_manual_output.view(3, 4, 7)
        assert_almost_equal(encoder_output.data.numpy(), manual_output.data.numpy())
