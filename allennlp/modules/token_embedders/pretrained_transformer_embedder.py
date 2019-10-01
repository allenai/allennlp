from overrides import overrides
from pytorch_transformers.modeling_auto import AutoModel
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("pretrained_transformer")
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
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

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:  # type: ignore

        return self.transformer_model(token_ids)[0]
