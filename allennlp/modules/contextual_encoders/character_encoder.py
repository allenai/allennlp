from typing import Sequence, Dict

import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.modules.highway import Highway
from allennlp.modules.masked_layer_norm import MaskedLayerNorm

_DEFAULT_FILTERS = ((1, 4), (2, 8), (3, 16), (4, 32), (5, 64))
_VALID_PROJECTION_LOCATIONS = {'after_cnn', 'after_highway'}

class CharacterEncoder(torch.nn.Module):
    def __init__(self,
                 activation: str = 'relu',
                 embedding_dim: int = 4,
                 filters: Sequence[Sequence[int]] = _DEFAULT_FILTERS,
                 max_characters_per_token: int = 50,
                 num_characters: int = 262,
                 num_highway: int = 2,
                 projection_dim: int = 512,
                 projection_location: str = 'after_cnn',
                 l2_coef: float = 0.0001,
                 do_layer_norm: bool = False) -> None:
        super().__init__()

        if projection_location not in _VALID_PROJECTION_LOCATIONS:
            raise ConfigurationError(f"unknown projection location: {projection_location}")

        self.output_dim = projection_dim
        self._max_characters_per_token = max_characters_per_token
        self._num_characters = num_characters
        self._projection_location = projection_location
        self._l2_coef = l2_coef

        if activation == 'tanh':
            self._activation = torch.nn.functional.tanh
        elif activation == 'relu':
            self._activation = torch.nn.functional.relu
        else:
            raise ConfigurationError(f"unknown activation {activation}")

        # char embedding
        self._char_embedding = torch.nn.Embedding(num_characters, embedding_dim)
        torch.nn.init.uniform_(self._char_embedding.weight.data, a=-1, b=1)

        # Create the convolutions
        self._convolutions = torch.nn.ModuleList()
        for width, num in filters:
            conv = torch.nn.Conv1d(in_channels=embedding_dim,
                                   out_channels=num,
                                   kernel_size=width,
                                   bias=True)
            conv.weight.data.uniform_(-0.05, 0.05)
            conv.bias.data.fill_(0.0)
            self._convolutions.append(conv)

        # Create the highway layers
        num_filters = sum(num for _, num in filters)
        if projection_location == 'after_cnn':
            highway_dim = projection_dim
        else:
            # highway_dim is the number of cnn filters
            highway_dim = num_filters
        self._highways = Highway(highway_dim, num_highway, activation=torch.nn.functional.relu)
        for highway in self._highways._layers:   # pylint: disable=protected-access
            # highway is a linear layer for each highway layer
            # with fused W and b weights
            highway.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / highway_dim))
            highway.bias[:highway_dim].data.fill_(0.0)
            highway.bias[highway_dim:].data.fill_(2.0)

        # Projection layer: always num_filters -> projection_dim
        self._projection = torch.nn.Linear(num_filters, projection_dim, bias=True)
        self._projection.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / num_filters))
        self._projection.bias.data.fill_(0.0)

        # And add a layer norm
        if do_layer_norm:
            self._layer_norm = MaskedLayerNorm(self.output_dim, gamma0=0.1)
        else:
            self._layer_norm = None

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs:
            Shape ``(batch_size, sequence_length, max_characters_per_token)`` of character ids
            representing the current batch.
        Returns
        -------
        Dict with keys:
        ``'token_embedding'``:
            Shape ``(batch_size, sequence_length, embedding_dim)``
            tensor with context
            insensitive token representations.
        ``'mask'``:
            Shape ``(batch_size, sequence_length)`` long tensor with
            sequence mask.
        """
        char_id_mask = (inputs > 0).long()  # (batch_size, sequence_length, max_characters_per_token)
        mask = (char_id_mask.sum(dim=-1) > 0).long()  # (batch_size, sequence_length)

        # character_id embedding
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = self._char_embedding(inputs.view(-1, self._max_characters_per_token))

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = character_embedding.transpose(1, 2)

        convolutions = []
        for convolution in self._convolutions:
            convolved = convolution(character_embedding)

            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = self._activation(convolved)
            convolutions.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convolutions, dim=-1)

        if self._projection_location == 'after_cnn':
            token_embedding = self._projection(token_embedding)

        # apply the highway layers (batch_size * sequence_length, highway_dim)
        token_embedding = self._highways(token_embedding)

        if self._projection_location == 'after_highway':
            # final projection  (batch_size * sequence_length, embedding_dim)
            token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = inputs.size()

        token_embedding_reshaped = token_embedding.view(batch_size, sequence_length, -1)

        if self._layer_norm:
            token_embedding_reshaped = self._layer_norm(token_embedding_reshaped, mask)

        return {
                'mask': mask,
                'token_embedding': token_embedding_reshaped
        }
