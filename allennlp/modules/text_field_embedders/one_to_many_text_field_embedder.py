from typing import Dict, List

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TextFieldEmbedder.register("one_to_many")
class OneToManyTextFieldEmbedder(TextFieldEmbedder):
    """
    This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.
    In a :class:`OneToManyTextFieldEmbedder`, a single :class:`~allennlp.data.TokenIndexer` can
    have multiple associated :class:`~allennlp.modules.token_embedders.TokenEmbedder` 's .
    This is useful in the case that you wish to represent the same set of token ids in
    multiple ways - for instance, using multiple different word embeddings.

    As the data produced by a :class:`~allennlp.data.fields.TextField` is a dictionary mapping
    :class:`~allennlp.data.TokenIndexer` names to these representations, we take a list of
    ``TokenEmbedders`` for each name, describing the methods used to embed each one.
    Each ``TokenEmbedder`` then embeds its input, and the result is concatenated in an
    arbitrary order.
    """
    def __init__(self, token_embedders: Dict[str, List[TokenEmbedder]]) -> None:
        super(OneToManyTextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        for key, namespace_embedders in token_embedders.items():
            for index, embedder in enumerate(namespace_embedders):
                name = f'token_embedder_{key}_{index}'
                self.add_module(name, embedder)

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for namespace_embedders in self._token_embedders.values():
            for embedder in namespace_embedders:
                output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._token_embedders.keys() != text_field_input.keys():
            message = f"Mismatched token keys: {self._token_embedders.keys()}" \
                      f" and {text_field_input.keys()}"
            raise ConfigurationError(message)
        embedded_representations = []
        keys = sorted(text_field_input.keys())
        for key in keys:
            tensor = text_field_input[key]
            for embedder in self._token_embedders[key]:
                token_vectors = embedder(tensor)
                embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BasicTextFieldEmbedder':
        token_embedders = {}
        keys = list(params.keys())
        for key in keys:
            all_embedder_params = params.pop(key)
            if not isinstance(all_embedder_params, list):
                raise ConfigurationError("The embedders passed to a OneToManyTextFieldEmbedder "
                                         "must be lists (they can be of length 1). Edit your "
                                         "configuration file to contain lists of embedders per "
                                         "namespace.")
            token_embedders[key] = [TokenEmbedder.from_params(vocab, embedder_params)
                                    for embedder_params in all_embedder_params]
        params.assert_empty(cls.__name__)
        return cls(token_embedders)
