import torch

class RNNBase(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = False) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = True
        self.bidirectional = bidirectional
