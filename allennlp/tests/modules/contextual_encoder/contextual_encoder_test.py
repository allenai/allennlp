# pylint: disable=invalid-name,no-self-use
import torch
import numpy as np

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.contextual_encoder.character_encoder import CharacterEncoder
from allennlp.modules.contextual_encoder.contextual_encoder import (
        CharLevelContextualEncoder, TokenLevelContextualEncoder)
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders import Embedding

class TestContextualEncoder(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.character_encoder = CharacterEncoder(
                activation='relu',
                embedding_dim=4,
                filters=[[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                max_characters_per_token=50,
                num_characters=262,
                num_highway=2,
                projection_dim=16,
                projection_location='after_cnn'
        )

        lstm = torch.nn.LSTM(bidirectional=True,
                             num_layers=3,
                             input_size=16,
                             hidden_size=10,
                             batch_first=True)
        self.seq2seq = PytorchSeq2SeqWrapper(lstm)


    def test_char_level_contextual_encoder(self):
        cew = CharLevelContextualEncoder(
                encoder=self.seq2seq,
                character_encoder=self.character_encoder,
                num_layers=4,
                return_all_layers=False)

        character_ids = torch.from_numpy(np.random.randint(0, 262, size=(5, 6, 50)))

        result = cew(character_ids)

        mask = result['mask']
        output = result['output']
        token_embedding = result['token_embedding']

        # output of char_encoder
        assert tuple(token_embedding.shape) == (5, 6, 16)
        assert tuple(mask.shape) == (5, 6)

        # output of bidirectional LSTM
        assert tuple(output.shape) == (5, 6, 20)

    def test_token_level_contextual_encoder(self):
        token_embedder = Embedding(num_embeddings=50, embedding_dim=16)

        cew = TokenLevelContextualEncoder(encoder=self.seq2seq,
                                          token_embedder=token_embedder,
                                          num_layers=3,
                                          return_all_layers=False)

        token_ids = torch.from_numpy(np.random.randint(0, 50, size=(5, 6)))
        result = cew(token_ids)

        mask = result['mask']
        output = result['output']
        token_embedding = result['token_embedding']

        # output of embedding
        assert tuple(token_embedding.shape) == (5, 6, 16)
        assert tuple(mask.shape) == (5, 6)

        # output of bidirectional LSTM
        assert tuple(output.shape) == (5, 6, 20)
