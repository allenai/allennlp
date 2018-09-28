from typing import Dict, List, Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.masked_layer_norm import MaskedLayerNorm
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders import TokenEmbedder

@TextFieldEmbedder.register("contextual")
class ContextualTextFieldEmbedder(TextFieldEmbedder):
    def __init__(self,
                 token_embedders: Dict[str, TokenEmbedder],
                 contextualizer: Seq2SeqEncoder,
                 num_layers: int,
                 dropout: float = None,
                 embedding_layer_norm: Optional[MaskedLayerNorm] = None,
                 return_all_layers: bool = False,
                 remove_boundary_tokens: bool = True,
                 embedder_to_indexer_map: Dict[str, List[str]] = None,
                 allow_unmatched_keys: bool = False) -> None:
        super().__init__()

        for key, embedder in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)

        self._token_embedders = token_embedders
        self._contextualizer = contextualizer
        self._num_layers = num_layers
        self._dropout = dropout
        self._embedding_layer_norm = embedding_layer_norm
        self._return_all_layers = return_all_layers
        self._remove_boundary_tokens = remove_boundary_tokens

        self._embedder_to_indexer_map = embedder_to_indexer_map
        self._allow_unmatched_keys = allow_unmatched_keys

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input: Dict[str, torch.Tensor], num_wrapping_dims: int = 0) -> torch.Tensor:
        if self._token_embedders.keys() != text_field_input.keys():
            if not self._allow_unmatched_keys:
                message = "Mismatched token keys: %s and %s" % (str(self._token_embedders.keys()),
                                                                str(text_field_input.keys()))
                raise ConfigurationError(message)
        embedded_representations = []
        keys = sorted(self._token_embedders.keys())
        for key in keys:
            # If we pre-specified a mapping explictly, use that.
            if self._embedder_to_indexer_map is not None:
                tensors = [text_field_input[indexer_key] for
                           indexer_key in self._embedder_to_indexer_map[key]]
            else:
                # otherwise, we assume the mapping between indexers and embedders
                # is bijective and just use the key directly.
                tensors = [text_field_input[key]]
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, 'token_embedder_{}'.format(key))
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            token_vectors = embedder(*tensors)
            contextualized = self._contextualizer(token_vectors)

            # remove boundary tokens
            if self._remove_boundary_tokens:
                contextualized = contextualized[:, 1:-1]

            embedded_representations.append(contextualized)
        return torch.cat(embedded_representations, dim=-1)
