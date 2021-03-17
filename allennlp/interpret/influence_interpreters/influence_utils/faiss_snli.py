import torch
from overrides import overrides
from allennlp.data import TextFieldTensors
from .faiss_wrapper import FAISSWrapper


class FAISSSnliWrapper(FAISSWrapper):
    @overrides
    def extract_tokens_from_input(self, tokens: TextFieldTensors, label: torch.IntTensor = None):
        return tokens
