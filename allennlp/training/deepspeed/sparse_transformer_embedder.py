from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder

from deepspeed.ops.sparse_attention.sparse_attention_utils import SparseAttentionUtils

@TokenEmbedder.register('sparse_transformer')
class SparseTransformerEmbedder(PretrainedTransformerEmbedder):
    class __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transformer_model = SparseAttentionUtils.replace_model_self_attention_with_sparse_self_attention(self.transformer_model)