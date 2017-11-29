import json
from typing import Union, List, Dict

import torch
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.modules.token_embedders.elmo_token_embedder import ELMoTokenEmbedder
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.common.checks import ConfigurationError


class ElmoBiLm(torch.nn.Module):
    """
    Run a pre-trained bidirectional language model, outputing the activations at each
    layer for weighting together into an ELMo representation (with
    ``allennlp.modules.seq2seq_encoders.Elmo``).  This is a lower level class, useful
    for advanced uses, but most users should use ``allennlp.modules.seq2seq_encoders.Elmo``
    directly.

    Parameters
    ----------
    options_file : str
        ELMo JSON options file
    weight_file : str
        ELMo hdf5 weight file
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str) -> None:
        super(ElmoBiLm, self).__init__()

        self._token_embedder = ELMoTokenEmbedder(options_file, weight_file)
        self.add_module('elmo_token_embedder', self._token_embedder)

        with open(cached_path(options_file), 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ConfigurationError('We only support pretrained biLMs with residual connections')
        self._elmo_lstm = ElmoLstm(input_size=options['lstm']['projection_dim'],
                                hidden_size=options['lstm']['projection_dim'],
                                cell_size=options['lstm']['dim'],
                                num_layers=options['lstm']['n_layers'],
                                memory_cell_clip_value=options['lstm']['cell_clip'],
                                state_projection_clip_value=options['lstm']['proj_clip'])
        self._elmo_lstm._load_weights(weight_file)
        self.add_module('elmo_lstm', self._elmo_lstm)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.

        Returns
        -------
        Dict with keys:

        ``'activations'``: ``List[torch.autograd.Variable]``
            A list of activations at each layer of the network, each of shape
            ``(batch_size, timesteps + 2, embedding_dim)``
        ``'mask'``:  ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.

        Note that the output tensors all include additional special begin and end of sequence
        markers.
        """
        token_embedding = self._token_embedder(inputs)
        type_representation = token_embedding['token_embedding']
        mask = token_embedding['mask']
        lstm_outputs = self._elmo_lstm(type_representation, mask)

        # Prepare the output.  The first layer is duplicated.
        output_tensors = [
                torch.cat([type_representation, type_representation], dim=-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {
                'activations': output_tensors,
                'mask': mask,
        }
