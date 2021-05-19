from allennlp.common import FromParams

from allennlp.modules.transformer.activation_layer import ActivationLayer
from allennlp.modules.transformer.output_layer import OutputLayer
from allennlp.modules.transformer.bimodal_attention import BiModalAttention

from allennlp.modules.transformer.transformer_module import TransformerModule


class BiModalOutput(TransformerModule, FromParams):
    def __init__(
        self,
        hidden_size1: int,
        hidden_size2: int,
        combined_hidden_size: int,
        dropout1: float,
        dropout2: float,
    ):
        super().__init__()

        self.bert_output1 = OutputLayer(combined_hidden_size, hidden_size1, dropout1)
        self.bert_output2 = OutputLayer(combined_hidden_size, hidden_size2, dropout2)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        hidden_states1 = self.bert_output1(hidden_states1, input_tensor1)
        hidden_states2 = self.bert_output2(hidden_states2, input_tensor2)

        return hidden_states1, hidden_states2


class BiModalConnectionLayer(TransformerModule, FromParams):

    _pretrained_mapping = {"biAttention": "bimodal_attention", "biOutput": "bimodal_output"}

    def __init__(
        self,
        hidden_size1: int,
        hidden_size2: int,
        combined_hidden_size: int,
        intermediate_size1: int,
        intermediate_size2: int,
        num_attention_heads: int,
        dropout1: float,
        dropout2: float,
        activation: str,
    ):
        super().__init__()
        self.bimodal_attention = BiModalAttention(
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            combined_hidden_size=combined_hidden_size,
            num_attention_heads=num_attention_heads,
            dropout1=dropout1,
            dropout2=dropout2,
        )

        self.bimodal_output = BiModalOutput(
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            combined_hidden_size=combined_hidden_size,
            dropout1=dropout1,
            dropout2=dropout2,
        )

        self.intermediate1 = ActivationLayer(
            hidden_size=hidden_size1,
            intermediate_size=intermediate_size1,
            activation=activation,
        )
        self.output1 = OutputLayer(
            hidden_size=hidden_size1,
            input_size=intermediate_size1,
            dropout=dropout1,
        )

        self.intermediate2 = ActivationLayer(
            hidden_size=hidden_size2,
            intermediate_size=intermediate_size2,
            activation=activation,
        )
        self.output2 = OutputLayer(
            hidden_size=hidden_size2,
            input_size=intermediate_size2,
            dropout=dropout2,
        )

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        bi_output1, bi_output2 = self.bimodal_attention(
            input_tensor1,
            input_tensor2,
            attention_mask1,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.bimodal_output(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.intermediate1(attention_output1)
        layer_output1 = self.output1(intermediate_output1, attention_output1)

        intermediate_output2 = self.intermediate2(attention_output2)
        layer_output2 = self.output2(intermediate_output2, attention_output2)

        return layer_output1, layer_output2
