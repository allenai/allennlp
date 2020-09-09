from typing import Optional

import torch

from allennlp.common import FromParams


class TransformerEmbeddings(torch.nn.Module, FromParams):
    """
    Construct the embeddings from word, position and token_type embeddings.
    Details in the paper:
    [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al, 2019]
    (https://api.semanticscholar.org/CorpusID:52967399)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        dropout: float,
    ):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert (input_ids is None and input_embeds is not None) or (
            input_ids is not None and input_embeds is None
        )

        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
            input_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = input_embeds.size()[:-1]
            device = input_embeds.device

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
