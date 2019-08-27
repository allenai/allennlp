from overrides import overrides
from pytorch_transformers import GPT2Config, GPT2LMHeadModel
import torch

from allennlp.modules.language_model_heads.language_model_head import LanguageModelHead


@LanguageModelHead.register('gpt2')
class Gpt2LanguageModelHead(LanguageModelHead):
    """
    Loads just the LM head from ``pytorch_transformers.GPT2LMHeadModel``.  It was easiest to load
    the entire model before only pulling out the head, so this is a bit slower than it could be,
    but for practical use in a model, the few seconds of extra loading time is probably not a big
    deal.
    """
    def __init__(self, model_name: str) -> None:
        super().__init__()
        config = GPT2Config.from_pretrained(model_name)
        self.input_dim = config.hidden_size
        self.output_dim = config.vocab_size
        # TODO(mattg): It's possible that we could use some kind of cache like we have in
        # allennlp.modules.token_embedders.bert_token_embedder.PretrainedBertModel.  That way, we
        # would only load the GPT2 weights once.  Though, it's not clear how to do that here, as we
        # need to load `GPT2LMHeadModel`, not just `GPT2Model`...
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt2_lm_head = gpt2_model.lm_head  # pylint: disable=no-member

    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.gpt2_lm_head(hidden_states)
