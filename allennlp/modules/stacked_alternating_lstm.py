import torch
from torch.nn.utils.rnn import PackedSequence
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.common.checks import ConfigurationError


class StackedAlternatingLstm(torch.nn.Module):
    """
    A stacked LSTM with LSTM layers which alternate between going forwards over
    the sequence and going backwards. This implementation is based on the
    description in `Deep Semantic Role Labelling - What works and what's next
    <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    Parameters
    ----------
    input_size : int, required
        The dimension of the inputs to the LSTM.
    hidden_size : int, required
        The dimension of the outputs of the LSTM.
    num_layers : int, required
        The number of stacked LSTMs to use.
    recurrent_dropout_probability: float, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .

    Return
    ------
    output_accumulator : PackedSequence
        The outputs of the interleaved LSTMs per timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 recurrent_dropout_probability: float = 0.0,
                 use_highway: bool = True) -> None:
        super(StackedAlternatingLstm, self).__init__()

        # Required to be wrapped with a :class:`PytorchSeq2SeqWrapper`.
        self.input_size = input_size
        self.hidden_size = hidden_size

        layers = []
        lstm_input_size = input_size
        for layer_index in range(num_layers):
            go_forward = True if layer_index % 2 == 0 else False
            layer = AugmentedLstm(lstm_input_size, hidden_size, go_forward,
                                  recurrent_dropout_probability=recurrent_dropout_probability,
                                  use_highway=use_highway)
            lstm_input_size = hidden_size
            self.add_module('layer_{}'.format(layer_index), layer)
            layers.append(layer)
        self.lstm_layers = layers

    def forward(self,
                inputs: PackedSequence,
                hidden_state: torch.Tensor = None):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        hidden_state : ``torch.Tensor``, optional, (default = None).
            A tensor of shape (num_layers, batch_size, hidden_size) to be used as the
            initial hidden state for the respective layers.

        Returns
        -------
        current : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size)
        final_states: torch.Tensor
            The per-layer final states of the LSTM, with shape (num_layers, batch_size, hidden_size).
        """
        if not hidden_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif hidden_state.size()[0] != len(self.lstm_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = hidden_state.split(1, 0)

        current = inputs
        final_states = []
        for layer, state in zip(self.lstm_layers, hidden_states):
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            if state is not None:
                state = (state, state)
            current, final_state = layer(current, state)
            final_states.append(final_state[0])
        return current, torch.cat(final_states, 0)
