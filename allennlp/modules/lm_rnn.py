from overrides import overrides

import torch
import torch.nn as nn
from torch import Tensor as Tensor
import torch.nn.functional as F
from torch.autograd import Variable

from allennlp.common.checks import check_dimensions_match
from allennlp.common.registrable import Registrable
from allennlp.common.params import Params

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class LMRNN(torch.nn.Module, Registrable):
    """
    Wrapper of RNN used in LMs
    """
    @overrides
    def forward(self, x) -> Tensor:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'LMRNN':
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
        
class VRNN_Basic(nn.Module):
    """
    One layer of the vanilla RNNs, used in vanilla stacked-RNNs

    Parameters
    ----------
    unit : ``str``, required
        Type of unit type
    input_dim : ``int``, required
        Size of hidden states
    dropout : ``float``, required
        The dropout ratio
    batch_norm : ``bool``, option (Default=``False``)
        Whether to include the batch norm in the feed forward direction
    """
    def __init__(self, unit: str, 
            input_dim: int, 
            hid_dim: int, 
            dropout: float, 
            batch_norm: bool = False) -> None:

        super(VRNN_Basic, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.layer = rnnunit_map[unit](input_dim, hid_dim, 1)
        self.dropout = dropout
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hid_dim)
        self.input_dim = input_dim
        self.output_dim = hid_dim

        self.init_hidden()

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_last_hidden(self) -> tuple:
        return self.hidden_state

    def init_hidden(self) -> None:

        self.hidden_state = None

    def forward(self, x) -> Tensor:

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        out, new_hidden = self.layer(x, self.hidden_state)

        self.hidden_state = repackage_hidden(new_hidden)

        if self.batch_norm:
            output_size = out.size()
            out = self.bn(out.view(-1, self.output_dim)).view(output_size)
        
        return out


@LMRNN.register("vanilla_RNN")
class vanilla_RNN(LMRNN):
    """
    vanilla stacked-RNNs

    Parameters
    ----------
    layer_num : ``int``, required
        number of RNN layers
    unit : ``str``, required
        Type of unit type
    input_dim : ``int``, required
        Size of hidden states
    dropout : ``float``, required
        The dropout ratio
    batch_norm : ``bool``, option (Default=``False``)
        Whether to include the batch norm in the feed forward direction
    """

    def __init__(self, layer_num:int, 
            unit:str, 
            input_dim:int, 
            hid_dim:int, 
            dropout:float, 
            batch_norm: bool = False) -> None:

        super(vanilla_RNN, self).__init__()

        layer_list = [VRNN_Basic(unit, input_dim, hid_dim, dropout, batch_norm)] + [VRNN_Basic(unit, hid_dim, hid_dim, dropout, batch_norm) for i in range(layer_num - 1)]
        self.layer = nn.Sequential(*layer_list)
        self.input_dim = layer_list[0].get_input_dim()
        self.output_dim = layer_list[-1].get_output_dim()

        self.init_hidden()

    @classmethod
    def from_params(cls, params: Params) -> 'LMRNN':

        layer_num = params.pop("layer_num")
        unit = params.pop("unit")
        input_dim = params.pop("input_dim")
        hid_dim = params.pop("hid_dim")
        dropout = params.pop("dropout")
        batch_norm = params.pop("batch_norm", False)

        return cls(layer_num=layer_num, 
            unit=unit, 
            input_dim=input_dim, 
            hid_dim=hid_dim, 
            dropout=dropout, 
            batch_norm=batch_norm)

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def init_hidden(self) -> None:
        for tup in self.layer.children():
            tup.init_hidden()

    def get_last_hidden(self) -> tuple:
        return [tup.get_last_hidden() for tup in self.layer.children()]

    def forward(self, x) -> Tensor:
        return self.layer(x)