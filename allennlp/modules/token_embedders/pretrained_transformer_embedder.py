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

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.transformer_model(token_ids)[0]
