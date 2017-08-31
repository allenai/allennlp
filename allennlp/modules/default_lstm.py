import re
from overrides import overrides
import torch

from allennlp.nn.initializers import block_orthogonal


class DefaultLstm(torch.nn.LSTM):
    """
    A wrapper of the Pytorch LSTM which provides a good default initialization
    of weights and biases, and disables training of the superfluous bias_hh parameters.

    Input parameters are identical to Pytorch LSTM.
    """
    @overrides
    def reset_parameters(self):
        hidden_size = self.hidden_size
        for param_name, param in self.named_parameters():
            if re.search("bias_hh", param_name):
                torch.nn.init.constant(param, 0)
                # This parameter is superfluous, so we should not update it in training
                param.requires_grad = False
            elif re.search("bias_ih", param_name):
                bias = param.data
                bias_size = bias.size(0)
                # The forget get bias is between fractions 1/4 and 1/2 of the vector
                start, end = bias_size//4, bias_size//2
                bias.fill_(0)
                bias[start:end].fill_(1)
            elif re.search("weight_ih", param_name):
                # Initialize each block matrix separately
                for i in range(0, 4):
                    block = param.data[i*hidden_size:(i+1)*hidden_size]
                    torch.nn.init.xavier_uniform(block)
            elif re.search("weight_hh", param_name):
                block_orthogonal(param, [hidden_size, hidden_size])
