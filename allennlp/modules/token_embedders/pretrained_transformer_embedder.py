from typing import Optional

from overrides import overrides
from transformers.modeling_auto import AutoModel
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("pretrained_transformer")
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

    @overrides
    def get_output_dim(self):
        return self.output_dim

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.LongTensor,
        type_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: torch.LongTensor
            Shape: [batch_size, num_wordpieces].
        mask: torch.LongTensor
            Shape: [batch_size, num_wordpieces].

        # Returns:

        Shape: [batch_size, num_wordpieces, embedding_size].
        """
        if (
            type_ids is not None
            and type_ids.max()
            >= self.transformer_model.embeddings.token_type_embeddings.num_embeddings
            == 1
        ):
            raise ValueError("Found type ids too large for the chosen transformer model.")
        return self.transformer_model(
            input_ids=token_ids, token_type_ids=type_ids, attention_mask=mask
        )[0]
