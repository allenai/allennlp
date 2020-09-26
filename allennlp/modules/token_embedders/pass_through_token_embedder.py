import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("pass_through")
class PassThroughTokenEmbedder(TokenEmbedder):
    """
    Assumes that the input is already vectorized in some way,
    and just returns it.

    Registered as a `TokenEmbedder` with name "pass_through".

    # Parameters

    hidden_dim : `int`, required.

    """

    def __init__(self, hidden_dim: int) -> None:
        self.hidden_dim = hidden_dim
        super().__init__()

    def get_output_dim(self):
        return self.hidden_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens
