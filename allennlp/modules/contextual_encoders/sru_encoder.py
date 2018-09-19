from typing import List, Dict

import torch

from allennlp.modules.contextual_encoders.contextual_encoder import ContextualEncoder

def reverse_padded_sequence(inputs: torch.Tensor,
                            lengths: List[int],
                            batch_first: bool = False) -> torch.Tensor:
    """
    Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).

    Parameters
    ----------
    inputs: ``torch.Tensor``
        A padded batch of variable-length sequences.
        The dimension of the vector for each element in the input sequence;
        ``input_tensor.size(-1)``.
    lengths: ``List[int]``
        The list of sequence lengths.
    batch_first: ``bool``, optional (default: False)
        If true, inputs are (batch_size, sequence_length, ...).
        Otherwise, inputs are (sequence_length, batch_size, ...).

    Returns
    -------
    a ``torch.Tensor`` with the same size as the inputs, but with each
    sequence reversed.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)

    max_length, batch_size, *_ = inputs.size()

    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')

    # [length - 1, length - 2, ..., 0, length, length + 1, ...]
    # shape (batch_size, max_length)
    indices = [list(reversed(range(0, length))) + list(range(length, max_length))
               for length in lengths]

    # shape (max_length, batch_size)
    indices = torch.LongTensor(indices, device=inputs.device).transpose(0, 1)

    # shape (max_length, batch_size, 1, 1, ..., 1)
    for dim in range(2, inputs.dim()):
        indices = indices.unsqueeze(dim)

    indices = indices.expand_as(inputs)

    # (max_length, batch_size, ...)
    reversed_inputs = torch.gather(inputs, 0, indices)

    if batch_first:
        # (batch_size, max_length)
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

@ContextualEncoder.register('sru-encoder')
class SruEncoder(ContextualEncoder):
    """
    ``ContextualEncoder`` that uses a Simple Recurrent Unit:
        https://github.com/taolei87/sru

    Notes on dropout: SRUCell uses two types of dropout:
    rnn_dropout and dropout. Both are variational (one mask per
    time series).  rnn_dropout is applied everywhere to input weights,
    dropout to output h vectors.
    rnn_dropout is applied to all layers, and dropout
    to all layers except the top.
    The PTB SRU model uses 6 layers, dim=910, dropout=rnn_dropout=0.2,
    and input embedding/softmax dropout=0.7

    Parameters
    ----------
    dim : int, optional (default = 32)
    input_dim : int, optional (default = None)
        If not specified, same as dim.
    num_layers : int, optional (default = 3)
        The number of SruCell layers.
    input_dropout : float, optional (default = 0.7)
        The dropout applied to the inputs.
    dropout : float, optional (default = 0.2)
        The dropout for all the SruCell layers but the last.
    rnn_dropout : float, optional (default = 0.2)
        The rnn_dropout for all the SruCell layers.
    l2_coef : float, optional (default = 0.0001)
        Coefficient for the regularization penalty.
    bias : float, optional (default = 0.0)
        Bias to apply to each SruCell.
    use_tanh : bool, optional (default = False)
        Passed to each SruCell.
    use_relu : bool, optional (default = False)
        Passed to each SruCell.
    """
    def __init__(self,
                 dim: int = 32,
                 input_dim: int = None,
                 num_layers: int = 3,
                 input_dropout: float = 0.7,
                 dropout: float = 0.2,
                 rnn_dropout: float = 0.2,
                 l2_coef: float = 0.0001,
                 bias: float = 0.0,
                 use_tanh: bool = False,
                 use_relu: bool = False) -> None:
        super().__init__(num_layers=num_layers,
                         output_dim=2 * dim)

        from sru.cuda_functional import SRUCell

        n_in = dim
        n_out = n_in

        first_layer_dim = input_dim or n_in

        self.forward_cells = torch.nn.ModuleList()
        self.backward_cells = torch.nn.ModuleList()

        cells = torch.nn.ModuleDict()
        for direction in ['forward', 'backward']:
            cells[direction] = torch.nn.ModuleList()
            for k in range(num_layers):
                cell = SRUCell(n_in=first_layer_dim if k == 0 else n_in,
                               n_out=n_out,
                               dropout=dropout if k < num_layers - 1 else 0,
                               rnn_dropout=rnn_dropout,
                               bidirectional=False,
                               use_tanh=int(use_tanh),
                               use_relu=int(use_relu))
                cell.set_bias(bias)
                cells[direction].append(cell)

        self._cells = cells

        self._direction_dim = n_out
        self._dropout = torch.nn.Dropout(input_dropout)
        self._l2_coef = l2_coef

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # token_embeddings = (batch_size, timesteps, dim)
        # SRU needs (timesteps, batch_size, dim)
        transposed_embeddings = torch.transpose(token_embeddings, 1, 0)

        # Make the initial hidden states
        batch_size = transposed_embeddings.size(1)
        c0 = token_embeddings.data.new_zeros(batch_size, self._direction_dim)  # pylint: disable=invalid-name

        # For forward direction, we assume ids start at index 0
        out = self._dropout(transposed_embeddings)
        for forward_cell in self._cells['forward']:
            out, _ = forward_cell(out, c0)
        # (batch_size, timesteps, forward_dim)
        forward_out = torch.transpose(out, 1, 0)

        # Now for backward embeddings
        # to deal with padding, we reverse sequences,
        # run a forward SRU, then undo.
        lengths = mask.long().sum(dim=1).detach().to(torch.device('cpu'))
        reversed_embeddings = reverse_padded_sequence(transposed_embeddings, lengths)
        out = self._dropout(reversed_embeddings)
        for backward_cell in self._cells['backward']:
            out, _ = backward_cell(out, c0)
        backward_out = torch.transpose(reverse_padded_sequence(out, lengths), 1, 0)

        return torch.cat([forward_out, backward_out], dim=2)

    def get_regularization_penalty(self) -> torch.Tensor:
        penalty = 0.0

        if self._l2_coef:
            for direction in ['forward', 'backward']:
                for cell in self._cells[direction]:
                    penalty += (cell.weight * cell.weight).sum()
            penalty *= self._l2_coef

        return penalty
