from .Module import Module

class RNNBase(Module):
    def __init__(self,
                 mode: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.,
                 bidirectional: bool = False) -> None: ...

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool

class Linear(Module):
    pass

class Dropout(Module):
    pass
