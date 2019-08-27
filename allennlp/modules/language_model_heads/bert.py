from overrides import overrides
from pytorch_transformers import BertConfig, BertForMaskedLM
import torch

from allennlp.modules.language_model_heads.language_model_head import LanguageModelHead


@LanguageModelHead.register('bert')
class BertLanguageModelHead(LanguageModelHead):
    """
    Loads just the LM head from ``pytorch_transformers.BertForMaskedLM``.  It was easiest to load
    the entire model before only pulling out the head, so this is a bit slower than it could be,
    but for practical use in a model, the few seconds of extra loading time is probably not a big
    deal.
    """
    def __init__(self, model_name: str) -> None:
        super().__init__()
        config = BertConfig.from_pretrained(model_name)
        self.input_dim = config.hidden_size
        self.output_dim = config.vocab_size
        # TODO(mattg): It's possible that we could use some kind of cache like we have in
        # allennlp.modules.token_embedders.bert_token_embedder.PretrainedBertModel.  That way, we
        # would only load the BERT weights once.  Though, it's not clear how to do that here, as we
        # need to load `BertForMaskedLM`, not just `BertModel`...
        bert_model = BertForMaskedLM.from_pretrained(model_name)
        self.bert_lm_head = bert_model.cls  # pylint: disable=no-member

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.bert_lm_head(hidden_states)
