from typing import Optional, Callable, Dict

import torch

from allennlp.common import Registrable
from allennlp.modules.contextual_encoders.character_encoder import CharacterEncoder
from allennlp.modules.contextual_encoders.contextual_encoder import ContextualEncoder
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.token_embedders import TokenEmbedder

class ContextualEncoderWrapper(torch.nn.Module, Registrable):
    """
    A ``ContextualEncoderWrapper`` encodes its inputs,
    uses a ``ContextualEncoder`` to add context to the encodings,
    then returns the results. This class is abstract; there are
    concrete subclasses for using character-level encodings
    or word embeddings.

    Parameters
    ----------
    contextual_encoder : ``ContextualEncoder``
        The ``ContextualEncoder`` to wrap.
    return_all_layers : bool, optional (default: False)
        Should this module return all layers or only the last layer?
    """
    def __init__(self,
                 contextual_encoder: ContextualEncoder,
                 return_all_layers: bool = False) -> None:
        super().__init__()
        self._contextual_encoder = contextual_encoder
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
    """
    A ``ContextualEncoderWrapper`` that uses a ``CharacterEncoder`` on its inputs.

    Parameters
    ----------
    contextual_encoder : ``ContextualEncoder``
        The ``ContextualEncoder`` to wrap.
    character_encoder : ``CharacterEncoder``
        The ``CharacterEncoder`` to apply to the inputs.
    return_all_layers : bool, optional (default: False)
        Should this module return all layers or only the last layer?
    """
    def __init__(self,
                 contextual_encoder: ContextualEncoder,
                 character_encoder: CharacterEncoder,
                 return_all_layers: bool = False) -> None:
        super().__init__(contextual_encoder, return_all_layers)
        self._character_encoder = character_encoder

    def forward(self, ids: torch.Tensor, callback: Callable = None) -> Dict[str, torch.Tensor]:
        encoder_output = self._character_encoder(ids)
        return self._forward_on_encoder_output(encoder_output['token_embedding'], encoder_output['mask'], callback)

    def get_regularization_penalty(self) -> torch.Tensor:
        return (self._contextual_encoder.get_regularization_penalty() +
                self._character_encoder.get_regularization_penalty())


@ContextualEncoderWrapper.register('token-level')
class TokenLevelContextualEncoderWrapper(ContextualEncoderWrapper):
    """
    A ``ContextualEncoderWrapper`` that applies a ``TokenEmbedder``
    to its inputs.

    Parameters
    ----------
    contextual_encoder : ``ContextualEncoder``
        The ``ContextualEncoder`` to wrap.
    token_embedder : ``TokenEmbedder``
        Used to embed the input tokens.
    embedding_layer_norm : ``MaskedLayerNorm``, optional (default: None)
        If supplied, this layer norm is applied to the token embeddings
        before passing them to the contextual encoder.
    return_all_layers : bool, optional (default: False)
        Should this module return all layers or only the last layer?
    """
    def __init__(self,
                 contextual_encoder: ContextualEncoder,
                 token_embedder: TokenEmbedder,
                 embedding_layer_norm: Optional[MaskedLayerNorm] = None,
                 return_all_layers: bool = False) -> None:
        super().__init__(contextual_encoder, return_all_layers)
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
