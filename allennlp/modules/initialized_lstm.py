from overrides import overrides
import torch

from allennlp.nn.initializers import block_orthogonal


class InitializedLstm(torch.nn.LSTM):
    """
    A wrapper of the Pytorch LSTM which provides a good default initialization
    of weights and biases, and disables training of the superfluous bias_hh parameters.

    Warning: Parameters with `param.requires_grad` set to `False` should not be passed
    to the optimizer.

    Input parameters are identical to Pytorch LSTM.
    """
    @overrides
    def reset_parameters(self):
        hidden_size = self.hidden_size
        for param_name, param in self.named_parameters():
            if param_name.startswith("bias_hh"):
                torch.nn.init.constant(param, 0)
                # This parameter is superfluous, so we should not update it in training
                param.requires_grad = False
            elif param_name.startswith("bias_ih"):
                bias = param.data
                # The forget get bias is in the second hidden_size block
                bias[hidden_size:2*hidden_size].fill_(1)
            elif param_name.startswith("weight_ih"):
                # Initialize each block matrix separately
                for i in range(0, 4):
                    block = param.data[i*hidden_size:(i+1)*hidden_size]
                    torch.nn.init.xavier_uniform(block)
            elif param_name.startswith("weight_hh"):
                block_orthogonal(param, [hidden_size, hidden_size])
