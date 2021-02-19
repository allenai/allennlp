from typing import Optional
from overrides import overrides

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from deepspeed.ops.sparse_attention import SparseAttentionUtils

from .sparse_attention import _SparsityConfig, replace_self_attention

import torch


@TokenEmbedder.register("sparse_transformer")
class SparseTransformerEmbedder(PretrainedTransformerEmbedder):
    def __init__(
        self,
        model_name: str,
        sparsity_config: _SparsityConfig = _SparsityConfig(num_heads=4),
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        self._sparsity_config = sparsity_config
        self.transformer_model = replace_self_attention(
            self.transformer_model, self._sparsity_config
        )

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore

        _, token_ids, mask, type_ids, *_ = SparseAttentionUtils.pad_to_block_size(
            block_size=self._sparsity_config.block,
            input_ids=token_ids,
            attention_mask=mask,
            token_type_ids=type_ids,
            position_ids=None,
            inputs_embeds=None,
            pad_token_id=self.transformer_model.config.pad_token_id,
            model_mbeddings=None,  # typo is in function definition, not here
        )
        return super().forward(
            token_ids=token_ids,
            mask=mask,
            type_ids=type_ids,
            segment_concat_mask=segment_concat_mask,
        )
