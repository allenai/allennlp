# pylint: disable=no-self-use
import numpy as np
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.contextual_encoders.character_encoder import CharacterEncoder

class TestCharacterEncoder(AllenNlpTestCase):
    def test_character_encoder(self):
        encoder = CharacterEncoder(
                activation='relu',
                embedding_dim=4,
                filters=[[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                max_characters_per_token=50,
                num_characters=262,
                num_highway=2,
                projection_dim=16,
                projection_location='after_cnn'
        )

        character_ids = torch.from_numpy(np.random.randint(0, 262, size=(5, 6, 50)))
        token_embedding = encoder(character_ids)

        assert list(token_embedding['token_embedding'].size()) == [5, 6, 16]
