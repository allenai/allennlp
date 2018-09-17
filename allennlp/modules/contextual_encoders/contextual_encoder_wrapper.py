from typing import Optional, Callable, Dict

import torch

from allennlp.common import Registrable
from allennlp.modules.contextual_encoders.character_encoder import CharacterEncoder
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.token_embedders import TokenEmbedder

class ContextualEncoderWrapper(torch.nn.Module, Registrable):
    def __init__(self,
                 contextual_encoder: torch.nn.Module,
                 encoder: Seq2SeqEncoder,
                 return_all_layers: bool = False) -> None:
        super().__init__()
        self._contextual_encoder = contextual_encoder
        self._encoder = encoder
        self._character_encoder = encoder
        self.num_layers = contextual_encoder.num_layers + 1
        self.output_dim = contextual_encoder.output_dim
        self.return_all_layers = return_all_layers

    def _forward_on_encoder_output(self,
                                   token_embedding: torch.Tensor,
                                   mask: torch.Tensor,
                                   callback: Callable) -> Dict[str, torch.Tensor]:
        if callback:
            token_embedding = callback(token_embedding)

        contextual_output = self._contextual_encoder(token_embedding, mask)

        if self.return_all_layers:
            # Concatenate all layers with the token layer
            token_layer = torch.cat([token_embedding, token_embedding], dim=-1)
            contextual_layers = torch.cat(
                    [layer.unsqueeze(1) for layer in contextual_output], dim=-1
            )
            contextual_output = torch.cat(
                    [token_layer.unsqueeze(1), contextual_layers], dim=-1
            )

        return {'output': contextual_output,
                'mask': mask,
                'token_embedding': token_embedding}

    def forward(self, ids: torch.Tensor, callback: Callable = None) -> Dict[str, torch.Tensor]:
        """
        If return_all_layers is True, returns {'output': (batch_size, num_layers, timesteps, dim)}
        Otherwise, returns {'output'}: (batch_size, timesteps, dim)
        """
        # pylint: disable=arguments-differ
        raise NotImplementedError

    def get_regularization_penalty(self) -> torch.Tensor:
        raise NotImplementedError

@ContextualEncoderWrapper.register('char-level')
class CharLevelContextualEncoderWrapper(ContextualEncoderWrapper):
    def __init__(self,
                 contextual_encoder: torch.nn.Module,
                 character_encoder: CharacterEncoder,
                 encoder: Seq2SeqEncoder,
                 return_all_layers: bool = False) -> None:
        super().__init__(contextual_encoder, encoder, return_all_layers)
        self._character_encoder = character_encoder

    def forward(self, ids: torch.Tensor, callback: Callable = None) -> Dict[str, torch.Tensor]:
        encoder_output = self._character_encoder(ids)
        return self._forward_on_encoder_output(encoder_output['token_embedding'], encoder_output['mask'], callback)

    def get_regularization_penalty(self) -> torch.Tensor:
        return (self._contextual_encoder.get_regularization_penalty() +
                self._character_encoder.get_regularization_penalty())


@ContextualEncoderWrapper.register('token-level')
class TokenLevelContextualEncoderWrapper(ContextualEncoderWrapper):
    def __init__(self,
                 contextual_encoder: torch.nn.Module,
                 token_embedder: TokenEmbedder,
                 encoder: Seq2SeqEncoder,
                 embedding_layer_norm: Optional[MaskedLayerNorm] = None,
                 return_all_layers: bool = False) -> None:
        super().__init__(contextual_encoder, encoder, return_all_layers)
        self._token_embedder = token_embedder
        self._embedding_layer_norm = embedding_layer_norm

    def forward(self, ids: torch.Tensor, callback: Callable = None) -> Dict[str, torch.Tensor]:
        mask = (ids > 0).long()
        embeddings = self._token_embedder(ids)

        if self._embedding_layer_norm:
            embeddings = self._embedding_layer_norm(embeddings, mask)

        return self._forward_on_encoder_output(embeddings, mask, callback)

    def get_regularization_penalty(self) -> torch.Tensor:
        return self._contextual_encoder.get_regularization_penalty()
