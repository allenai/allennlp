from typing import Optional, Callable, Dict

import torch

from allennlp.common import Registrable
from allennlp.modules.contextual_encoder.character_encoder import CharacterEncoder
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.token_embedders import TokenEmbedder

class ContextualEncoder(torch.nn.Module, Registrable):
    """
    A ``ContextualEncoder`` encodes its inputs,
    uses a ``Seq2SeqEncoder`` to add context to the encodings,
    then returns the results. This class is abstract; there are
    concrete subclasses for using character-level encodings
    or word embeddings.

    Parameters
    ----------
    encoder : ``Seq2SeqEncoder``
        The ``Seq2SeqEncoder`` to wrap.
    return_all_layers : bool, optional (default: False)
        Should this module return all layers or only the last layer?
    """
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 num_layers: int,
                 dropout: float = None,
                 return_all_layers: bool = False) -> None:
        super().__init__()
        self._encoder = encoder
        self.num_layers = num_layers
        self.output_dim = encoder.get_output_dim()
        self.return_all_layers = return_all_layers

        if dropout is not None:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

    def _forward_on_encoder_output(self,
                                   token_embedding: torch.Tensor,
                                   mask: torch.Tensor,
                                   callback: Callable) -> Dict[str, torch.Tensor]:
        if callback:
            token_embedding = callback(token_embedding)

        if self._dropout:
            token_embedding = self._dropout(token_embedding)

        contextual_output = self._encoder(token_embedding, mask)

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


@ContextualEncoder.register('char-level')
class CharLevelContextualEncoder(ContextualEncoder):
    """
    A ``ContextualEncoder`` that uses a ``CharacterEncoder`` on its inputs.

    Parameters
    ----------
    encoder : ``Seq2SeqEncoder``
        The ``Seq2SeqEncoder`` to wrap.
    character_encoder : ``CharacterEncoder``
        The ``CharacterEncoder`` to apply to the inputs.
    return_all_layers : bool, optional (default: False)
        Should this module return all layers or only the last layer?
    """
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 character_encoder: CharacterEncoder,
                 num_layers: int,
                 dropout: float = None,
                 return_all_layers: bool = False) -> None:
        super().__init__(encoder, num_layers, dropout, return_all_layers)
        self._character_encoder = character_encoder

    def forward(self, ids: torch.Tensor, callback: Callable = None) -> Dict[str, torch.Tensor]:
        encoder_output = self._character_encoder(ids)
        return self._forward_on_encoder_output(encoder_output['token_embedding'], encoder_output['mask'], callback)


@ContextualEncoder.register('token-level')
class TokenLevelContextualEncoder(ContextualEncoder):
    """
    A ``ContextualEncoder`` that applies a ``TokenEmbedder`` to its inputs.

    Parameters
    ----------
    encoder : ``Seq2SeqEncoder``
        The ``Seq2SeqEncoder`` to wrap.
    token_embedder : ``TokenEmbedder``
        Used to embed the input tokens.
    embedding_layer_norm : ``MaskedLayerNorm``, optional (default: None)
        If supplied, this layer norm is applied to the token embeddings
        before passing them to the contextual encoder.
    return_all_layers : bool, optional (default: False)
        Should this module return all layers or only the last layer?
    """
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 token_embedder: TokenEmbedder,
                 num_layers: int,
                 dropout: float = None,
                 embedding_layer_norm: Optional[MaskedLayerNorm] = None,
                 return_all_layers: bool = False) -> None:
        super().__init__(encoder, num_layers, dropout, return_all_layers)
        self._token_embedder = token_embedder
        self._embedding_layer_norm = embedding_layer_norm

    def forward(self, ids: torch.Tensor, callback: Callable = None) -> Dict[str, torch.Tensor]:
        mask = (ids > 0).long()
        embeddings = self._token_embedder(ids)

        if self._embedding_layer_norm:
            embeddings = self._embedding_layer_norm(embeddings, mask)

        return self._forward_on_encoder_output(embeddings, mask, callback)
