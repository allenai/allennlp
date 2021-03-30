from allennlp.common import FromParams
from allennlp.modules.transformer.activation_layer import ActivationLayer


class TransformerPooler(ActivationLayer, FromParams):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__(hidden_size, intermediate_size, "relu", pool=True)
