import torch


class RMSNorm(nn.Module):
    """
    An implementation of [RMS Normalization](
    https://openreview.net/pdf?id=SygkZ3MTJE.
    RMS Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:
    output = (gain * (tensor / (std + epsilon)) + bias
    # Parameters
    size : `int`, required.
    The dimension of the layer output to normalize.
    eps : `float`, optional, (default = 1e-8)
    An epsilon to prevent dividing by zero in the case
    the layer has zero variance.
    # Returns
    The normalized layer output.
    """  # noqa
    
    def __init__(self,size: int,eps: float = 1e-8):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(size))   ## Gain
        self.bias = torch.nn.Parameter(torch.zeros(size))  ## Bias
        self.epsilon = epsilon
    
    def forward(self,tensor:Tensor):
        std = torch.sqrt(torch.mean(tensor**2,-1,keepdim=True))    ## STD of Tensor along last dimension
        
        return  self.gain * (tensor /(std + self.epsilon)) + self.bias  ## Normalized Tensor
