from typing import Dict, List, Union, Any
import inspect

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TextFieldEmbedder.register("basic")
class BasicTextFieldEmbedder(TextFieldEmbedder):
    """
    This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.  Each
    ``TokenEmbedder`` embeds or encodes the representation output from one
    :class:`~allennlp.data.TokenIndexer`.  As the data produced by a
    :class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
    representations, we take ``TokenEmbedders`` with corresponding names.  Each ``TokenEmbedders``
    embeds its input, and the result is concatenated in an arbitrary (but consistent) order.

    # Parameters


    token_embedders : ``Dict[str, TokenEmbedder]``, required.
        A dictionary mapping token embedder names to implementations.
        These names should match the corresponding indexer used to generate
        the tensor passed to the TokenEmbedder.
    """

    def __init__(
        self,
        token_embedders: Dict[str, TokenEmbedder],
    ) -> None:
        super().__init__()
        # NOTE(mattg): I'd prefer to just use ModuleDict(token_embedders) here, but that changes
        # weights and invalidates all prior models, just for a cosmetic change in the code.
        self._token_embedders = token_embedders
        for key, embedder in token_embedders.items():
            name = "token_embedder_%s" % key
            self.add_module(name, embedder)
        self._ordered_embedder_keys = sorted(self._token_embedders.keys())

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(
        self, text_field_input: TextFieldTensors, num_wrapping_dims: int = 0, **kwargs
    ) -> torch.Tensor:
        embedder_keys = self._token_embedders.keys()
        input_keys = text_field_input.keys()

        if self._token_embedders.keys() != text_field_input.keys():
            message = "Mismatched token keys: %s and %s" % (
                str(self._token_embedders.keys()),
                str(text_field_input.keys()),
            )
            raise ConfigurationError(message)

        embedded_representations = []
        for key in self._ordered_embedder_keys:
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, "token_embedder_{}".format(key))
            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]

            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            else:
                tensors: Dict[str, torch.Tensor] = text_field_input[key]
                # TODO(mattg): this requires that dictionary keys in a TokenIndexer match the
                # argument names in a TokenEmbedder.  It might be nice to soften this requirement,
                # trying to auto-detect a mapping in some cases.  Not sure that it's worth the
                # complexity, though.
                token_vectors = embedder(**tensors, **forward_params_values)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)
