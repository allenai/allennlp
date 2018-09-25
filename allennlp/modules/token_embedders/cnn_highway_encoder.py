from typing import Sequence, Dict, List

import torch
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.modules.highway import Highway
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.nn.util import add_sentence_boundary_token_ids

_VALID_PROJECTION_LOCATIONS = {'after_cnn', 'after_highway', None}

@TokenEmbedder.register('cnn-highway')
class CnnHighwayEncoder(TokenEmbedder):
    """
    The character CNN + highway encoder from Kim et al "Character aware neural language models"
    https://arxiv.org/abs/1508.06615
    with an optional projection.

    Parameters
    ----------
    embedding_dim: int
        The dimension of the initial character embedding.
    filters: ``Sequence[Sequence[int]]``
        A sequence of pairs (filter_width, num_filters).
    num_highway: int
        The number of highway layers.
    projection_dim: int
        The output dimension of the projection layer.
    activation: str, optional (default = 'relu')
        The activation function for the convolutional layers.
    max_characters_per_token: int, optional (default = 50)
        The maximum length of any token.
    num_characters: int, optional (default = 262)
        The number of characters in the vocabulary.
    projection_location: str, optional (default = 'after_highway')
        Where to apply the projection layer. Valid values are
        'after_highway', 'after_cnn', and None.
    do_layer_norm: bool, optional (default = False)
        If True, we apply ``MaskedLayerNorm`` to the final encoded result.
    bos_characters: ``List[int]``, optional (default = None)
        Characters for the beginning-of-sentence token (if any).
        If provided, they will be prepended to each sentence.
        If provided, ``eos_characters`` must be provided as well.
    eos_characters: ``List[int]``, optional (default = None)
        Characters for the end-of-sentence token (if any).
        If provided, they will be appended to each sentence.
        If provided, ``bos_characters`` must be provided as well.
    return_mask: ``bool``, optional (default = False)
        If True, the forward method will return a dict containing
        both the embedding and the mask. (This is unusual ``TokenEmbedder``
        behavior, but the current ``Elmo`` model uses the ``CnnHighwayEncoder``
        internally and requires this dict to be returned.) Otherwise, it will just
        return the embedding tensor. (The ``TokenEmbedder`` contract
        is agnostic on this issue, but the ``BasicTextFieldEmbedder``
        requires its token embedders to return bare tensors.)
    """
    def __init__(self,
                 embedding_dim: int,
                 filters: Sequence[Sequence[int]],
                 num_highway: int,
                 projection_dim: int,
                 activation: str = 'relu',
                 max_characters_per_token: int = 50,
                 num_characters: int = 262,
                 projection_location: str = 'after_highway',
                 do_layer_norm: bool = False,
                 bos_characters: List[int] = None,
                 eos_characters: List[int] = None,
                 return_mask: bool = False) -> None:
        super().__init__()

        if projection_location not in _VALID_PROJECTION_LOCATIONS:
            raise ConfigurationError(f"unknown projection location: {projection_location}")

        self.output_dim = projection_dim
        self._max_characters_per_token = max_characters_per_token
        self._num_characters = num_characters
        self._projection_location = projection_location
        self._return_mask = return_mask

        if bos_characters and eos_characters:
            # Add 1 for masking.
            self._bos_characters = torch.from_numpy(np.array(bos_characters) + 1)
            self._eos_characters = torch.from_numpy(np.array(eos_characters) + 1)
        elif bos_characters or eos_characters:
            raise ConfigurationError("must specify both bos_characters and eos_characters or neither")
        else:
            self._bos_characters = None
            self._eos_characters = None

        if activation == 'tanh':
            self._activation = torch.nn.functional.tanh
        elif activation == 'relu':
            self._activation = torch.nn.functional.relu
        else:
            raise ConfigurationError(f"unknown activation {activation}")

        # char embedding
        weights = np.random.uniform(-1, 1, (num_characters, embedding_dim)).astype('float32')
        self._char_embedding_weights = torch.nn.Parameter(torch.FloatTensor(weights))

        # Create the convolutions
        self._convolutions: List[torch.nn.Module] = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(in_channels=embedding_dim,
                                   out_channels=num,
                                   kernel_size=width,
                                   bias=True)
            conv.weight.data.uniform_(-0.05, 0.05)
            conv.bias.data.fill_(0.0)
            self.add_module(f"char_conv_{i}", conv)  # needs to match the old ELMo name
            self._convolutions.append(conv)

        # Create the highway layers
        num_filters = sum(num for _, num in filters)
        if projection_location == 'after_cnn':
            highway_dim = projection_dim
        else:
            # highway_dim is the number of cnn filters
            highway_dim = num_filters
        self._highways = Highway(highway_dim, num_highway, activation=torch.nn.functional.relu)
        for highway_layer in self._highways._layers:   # pylint: disable=protected-access
            # highway is a linear layer for each highway layer
            # with fused W and b weights
            highway_layer.weight.data.normal_(mean=0.0, std=np.sqrt(1.0 / highway_dim))
            highway_layer.bias[:highway_dim].data.fill_(0.0)
            highway_layer.bias[highway_dim:].data.fill_(2.0)

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
        ``token_embedding``:
            Shape ``(batch_size, sequence_length, embedding_dim)`` tensor
            with context-insensitive token representations. If bos_characters and eos_characters
            are being added, the second dimension will be ``sequence_length + 2``.
        """
        # pylint: disable=arguments-differ
        char_id_mask = (inputs > 0).long()  # (batch_size, sequence_length, max_characters_per_token)
        mask = (char_id_mask.sum(dim=-1) > 0).long()  # (batch_size, sequence_length)

        # Add BOS / EOS
        if self._bos_characters is not None:
            inputs, mask = add_sentence_boundary_token_ids(inputs,
                                                           mask,
                                                           self._bos_characters, self._eos_characters)

        # character_id embedding
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(inputs.view(-1, self._max_characters_per_token),
                                                            self._char_embedding_weights)

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = character_embedding.transpose(1, 2)

        convolutions = []
        for i in range(len(self._convolutions)):
            char_conv_i = getattr(self, f"char_conv_{i}")
            convolved = char_conv_i(character_embedding)

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

        if self._return_mask:
            return {"mask": mask, "token_embedding": token_embedding_reshaped}
        else:
            return token_embedding_reshaped

    def get_output_dim(self) -> int:
        return self.output_dim
