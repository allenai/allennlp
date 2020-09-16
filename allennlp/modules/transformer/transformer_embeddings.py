from typing import Optional, List

import torch

from allennlp.common import FromParams

from allennlp.modules.transformer.transformer_module import TransformerModule


class Embeddings(TransformerModule, FromParams):
    """
    General class for embeddings for any modality.
    """

    def __init__(self, embeddings: List[torch.nn.Module], hidden_size: int, dropout: float):
        super().__init__()
        self.embeddings = embeddings
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == len(self.embeddings)
        outputs = []
        for i, inp in enumerate(inputs):
            outputs.append(self.embeddings[i](inp))

        outputs = sum(outputs)  # type: ignore
        outputs = self.layer_norm(outputs)
        outputs = self.dropout(outputs)
        return outputs


class TransformerEmbeddings(TransformerModule, FromParams):
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
        word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)

        embeddings = [word_embeddings, position_embeddings, token_type_embeddings]

        self.embeddings = Embeddings(embeddings, hidden_size, dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        input_shape = input_ids.size()
        device = input_ids.device
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embeddings = self.embeddings([input_ids, position_ids, token_type_ids])

        return embeddings
