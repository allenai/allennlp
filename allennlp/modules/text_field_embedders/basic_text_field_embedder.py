from typing import Dict

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
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
    embeds its input, and the result is concatenated in an arbitrary order.
    """
    def __init__(self, token_embedders: Dict[str, TokenEmbedder]) -> None:
        super(BasicTextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        for key, embedder in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input: Dict[str, torch.Tensor], num_wrapping_dims: int = 0) -> torch.Tensor:
        if self._token_embedders.keys() != text_field_input.keys():
            message = "Mismatched token keys: %s and %s" % (str(self._token_embedders.keys()),
                                                            str(text_field_input.keys()))
            raise ConfigurationError(message)
        embedded_representations = []
        keys = sorted(text_field_input.keys())
        for key in keys:
            tensor = text_field_input[key]
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, 'token_embedder_{}'.format(key))
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            token_vectors = embedder(tensor)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BasicTextFieldEmbedder':
        token_embedders = {}
        keys = list(params.keys())
        for key in keys:
            embedder_params = params.pop(key)
            token_embedders[key] = TokenEmbedder.from_params(vocab, embedder_params)
        params.assert_empty(cls.__name__)
        return cls(token_embedders)
