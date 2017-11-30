import json
from typing import Union, List, Dict

import torch
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common import Registrable, Params
from allennlp.modules.token_embedders.elmo_token_embedder import _ElmoTokenRepresentation
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules import ScalarMix
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.data import Vocabulary


@Registrable.register('elmo')
class Elmo(torch.nn.Module, Registrable):
    """
    Compute ELMo representations using a pre-trained bidirectional language model.

    See "Deep contextualized word representations", Peters et al. for details.

    This module takes character id input and computes ``num_elmo_layers`` different layers
    of ELMo representations.  Typically ``num_elmo_layers`` is 1 or 2.  For example, in
    the case of the SRL model in the above paper, ``num_elmo_layers=1`` where ELMo was included at
    the input token representation layer.  In the case of the SQuAD model, ``num_elmo_layers=2``
    as ELMo was also included at the GRU output layer.

    In the implementation below, we learn separate scalar weights for each output layer,
    but only run the biLM once on each input sequence for efficiency.

    Parameters
    ----------
    options_file : str
        ELMo JSON options file
    weight_file : str
        ELMo hdf5 weight file
    num_elmo_layers: int
        The number of ELMo representation layers to output.
    do_layer_norm: bool
        Should we apply layer normalization (passed to ``ScalarMix``)?
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 num_elmo_layers: int,
                 do_layer_norm: bool=False) -> None:
        super(Elmo, self).__init__()

        self._elmo_lstm = _ElmoBiLm(options_file, weight_file)
        #self.add_module('elmo_lstm', self._elmo_lstm)

        self._scalar_mixes = []
        for k in range(num_elmo_layers):
            scalar_mix = ScalarMix(self._elmo_lstm.num_layers, do_layer_norm=do_layer_norm)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def forward(self, inputs: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.

        Returns
        -------
        Dict with keys:

        ``'elmo'``: ``List[torch.autograd.Variable]``
            A ``num_elmo_layers`` list of ELMo representations for the input sequence.
            Each representation is shape ``(batch_size, timesteps, embedding_dim)``
        ``'mask'``:  ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        """
        bilm_output = self._elmo_lstm(inputs)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        elmo_representations = []
        for scalar_mix in self._scalar_mixes:
            representation_with_bos_eos = scalar_mix.forward(layer_activations, mask_with_bos_eos)
            representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                    representation_with_bos_eos, mask_with_bos_eos
            )
            elmo_representations.append(representation_without_bos_eos)

        return {'elmo': elmo_representations, 'mask': mask_without_bos_eos}

    @classmethod
    def from_params(cls, params: Params) -> 'Elmo':
        options_file = params.pop('options_file')
        weight_file = params.pop('weight_file')
        num_elmo_layers = params.pop('num_elmo_layers')
        do_layer_norm = params.pop('do_layer_norm', False)
        params.assert_empty(cls.__name__)
        return cls(options_file, weight_file, num_elmo_layers, do_layer_norm)


@TokenEmbedder.register("elmo_token_embedder")
class ElmoTokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of ELMo representations.

    This class servers as a convenience when you only want to use one layer of
    ELMo representations at the input of your network.  It's essentially a wrapper
    around Elmo(num_elmo_layers=1, ...)

    Parameters
    ----------
    options_file : str
        ELMo JSON options file
    weight_file : str
        ELMo hdf5 weight file
    do_layer_norm: bool
        Should we apply layer normalization (passed to ``ScalarMix``)?
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 do_layer_norm: bool=False) -> None:
        super(ElmoTokenEmbedder, self).__init__()

        self._elmo = Elmo(options_file, weight_file, 1, do_layer_norm=do_layer_norm)
        #self.add_module('elmo', self._elmo)

    def get_output_dim(self):
        return 2 * self._elmo._elmo_lstm._token_embedder.get_output_dim()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.

        Returns
        -------
        The ELMo representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        elmo_output = self._elmo(inputs)
        return elmo_output['elmo'][0]

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ElmoTokenEmbedder':
        options_file = params.pop('options_file')
        weight_file = params.pop('weight_file')
        do_layer_norm = params.pop('do_layer_norm', False)
        params.assert_empty(cls.__name__)
        return cls(options_file, weight_file, do_layer_norm)


class _ElmoBiLm(torch.nn.Module):
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
        super(_ElmoBiLm, self).__init__()

        self._token_embedder = _ElmoTokenRepresentation(options_file, weight_file)
        #self.add_module('elmo_token_embedder', self._token_embedder)

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
        #self.add_module('elmo_lstm', self._elmo_lstm)
        # Number of representation layers including context independent layer
        self.num_layers = options['lstm']['n_layers'] + 1

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
