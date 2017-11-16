"""
Compute the context insensitive token embeddings from pretrained biLMs.
"""
import json
import logging

from typing import Dict

import numpy
import h5py

from overrides import overrides

import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.highway import Highway
from allennlp.nn.util import add_sentence_boundary_token_ids
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
DTYPE = 'float32'

# pylint: disable=attribute-defined-outside-init


@TokenEmbedder.register("elmo_token_embedder")
class ELMoTokenEmbedder(TokenEmbedder):
    """
    Compute context sensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.

    Parameters
    ----------
    options_file : str
        ELMo JSON options file
    weight_file : str
        ELMo hdf5 weight file

    The relevant section of the options file is something like:
    .. example-code::

        .. code-block:: python

            {'char_cnn': {
                'activation': 'relu',
                'embedding': {'dim': 4},
                'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                'max_characters_per_token': 50,
                'n_characters': 262,
                'n_highway': 2
                }
            }

    something
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str) -> None:
        super(ELMoTokenEmbedder, self).__init__()

        with open(cached_path(options_file), 'r') as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file

        self.output_dim = self._options['lstm']['projection_dim']

        self._load_weights()

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs: ``torch.autograd.Variable``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.

        Returns
        -------
        Dict with keys:

        ``'token_embedding'``: ``torch.autograd.Variable``
            Shape ``(batch_size, sequence_length + 2, embedding_dim)`` tensor with context
            insensitive token representations.
        ``'mask'``:  ``torch.autograd.Variable``
            Shape ``(batch_size, sequence_length + 2)`` long tensor with sequence mask.
        """
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
                inputs,
                mask,
                Variable(torch.from_numpy(numpy.array(ELMoCharacterMapper.beginning_of_sentence_characters))),
                Variable(torch.from_numpy(numpy.array(ELMoCharacterMapper.end_of_sentence_characters)))
        )

        # the character id embedding
        max_chars_per_token = self._options['char_cnn']['max_characters_per_token']
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
                character_ids_with_bos_eos.view(-1, max_chars_per_token),
                self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = torch.nn.functional.tanh
        elif cnn_options['activation'] == 'relu':
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for conv in self._convolutions:
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return {
                'mask': mask_with_bos_eos,
                'token_embedding': token_embedding.view(batch_size, sequence_length, -1)
        }

    def _load_weights(self):
        self._load_char_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_char_embedding(self):
        with h5py.File(cached_path(self._weight_file), 'r') as fin:
            char_embed_weights = fin['char_embed'][...]

        weights = numpy.zeros(
                (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
                dtype=DTYPE
        )
        weights[1:, :] = char_embed_weights

        self._char_embedding_weights = torch.nn.Parameter(
                torch.FloatTensor(weights), requires_grad=False
        )

    def _load_cnn_weights(self):
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        char_embed_dim = cnn_options['embedding']['dim']

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                    in_channels=char_embed_dim,
                    out_channels=num,
                    kernel_size=width,
                    bias=True
            )
            # load the weights
            with h5py.File(cached_path(self._weight_file), 'r') as fin:
                weight = fin['CNN']['W_cnn_{}'.format(i)][...]
                bias = fin['CNN']['b_cnn_{}'.format(i)][...]

            w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError("Invalid weight file")
            conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
            conv.bias.data.copy_(torch.FloatTensor(bias))

            conv.weight.requires_grad = False
            conv.bias.requires_grad = False

            convolutions.append(conv)

        self._convolutions = convolutions

    def _load_highway(self):
        # pylint: disable=protected-access
        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options['n_highway']

        # create the layers, and load the weights
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multplication with concatenation of
            # transform and carry weights.
            with h5py.File(cached_path(self._weight_file), 'r') as fin:
                # The weights are transposed due to multiplication order assumptions in tf
                # vs pytorch (tf.matmul(X, W) vs pytorch.matmul(W, X))
                w_transform = numpy.transpose(fin['CNN_high_{}'.format(k)]['W_transform'][...])
                # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                w_carry = -1.0 * numpy.transpose(fin['CNN_high_{}'.format(k)]['W_carry'][...])
                weight = numpy.concatenate([w_transform, w_carry], axis=0)
                self._highways._layers[k].weight.data.copy_(torch.FloatTensor(weight))
                self._highways._layers[k].weight.requires_grad = False

                b_transform = fin['CNN_high_{}'.format(k)]['b_transform'][...]
                b_carry = -1.0 * fin['CNN_high_{}'.format(k)]['b_carry'][...]
                bias = numpy.concatenate([b_transform, b_carry], axis=0)
                self._highways._layers[k].bias.data.copy_(torch.FloatTensor(bias))
                self._highways._layers[k].bias.requires_grad = False

    def _load_projection(self):
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)

        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        with h5py.File(cached_path(self._weight_file), 'r') as fin:
            weight = fin['CNN_proj']['W_proj'][...]
            bias = fin['CNN_proj']['b_proj'][...]
            self._projection.weight.data.copy_(torch.FloatTensor(numpy.transpose(weight)))
            self._projection.bias.data.copy_(torch.FloatTensor(bias))

            self._projection.weight.requires_grad = False
            self._projection.bias.requires_grad = False

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ELMoTokenEmbedder':
        options_file = params.pop('options_file')
        weight_file = params.pop('weight_file')
        params.assert_empty(cls.__name__)
        return cls(options_file, weight_file)
