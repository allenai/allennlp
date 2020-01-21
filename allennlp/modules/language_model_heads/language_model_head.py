import torch

from allennlp.common import Registrable


class LanguageModelHead(torch.nn.Module, Registrable):
    """
    A `LanguageModelHead` encapsulates a function that goes from some hidden state to logits over
    a vocabulary.
    """

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore

        raise NotImplementedError
