from overrides import overrides
from pytorch_transformers import BertConfig, BertForMaskedLM
import torch

from allennlp.modules.language_model_heads.language_model_head import LanguageModelHead


@LanguageModelHead.register('bert')
class BertLanguageModelHead(LanguageModelHead):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.input_dim = config.hidden_size
        self.output_dim = config.vocab_size
        bert_model = BertForMaskedLM.from_pretrained(model_name)
        self.bert_lm_head = bert_model.cls

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.bert_lm_head(hidden_states)
