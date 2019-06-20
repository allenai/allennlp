from typing import Dict, List, Union, Any
import inspect

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

    Parameters
    ----------

    token_embedders : ``Dict[str, TokenEmbedder]``, required.
        A dictionary mapping token embedder names to implementations.
        These names should match the corresponding indexer used to generate
        the tensor passed to the TokenEmbedder.
    embedder_to_indexer_map : ``Dict[str, Union[List[str], Dict[str, str]]]``, optional, (default = None)
        Optionally, you can provide a mapping between the names of the TokenEmbedders that
        you are using to embed your TextField and an ordered list of indexer names which
        are needed for running it, or a mapping between the parameters which the
        ``TokenEmbedder.forward`` takes and the indexer names which are viewed as arguments.
        In most cases, your TokenEmbedder will only require a single tensor, because it is
        designed to run on the output of a single TokenIndexer. For example, the ELMo Token
        Embedder can be used in two modes, one of which requires both character ids and word
        ids for the same text. Note that the list of token indexer names is `ordered`,
        meaning that the tensors produced by the indexers will be passed to the embedders in
        the order you specify in this list. You can also use `null` in the configuration to
        set some specified parameters to None.
    allow_unmatched_keys : ``bool``, optional (default = False)
        If True, then don't enforce the keys of the ``text_field_input`` to
        match those in ``token_embedders`` (useful if the mapping is specified
        via ``embedder_to_indexer_map``).
    """
    def __init__(self,
                 token_embedders: Dict[str, TokenEmbedder],
                 embedder_to_indexer_map: Dict[str, Union[List[str], Dict[str, str]]] = None,
                 allow_unmatched_keys: bool = False) -> None:
        super(BasicTextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        self._embedder_to_indexer_map = embedder_to_indexer_map
        for key, embedder in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)
        self._allow_unmatched_keys = allow_unmatched_keys

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input: Dict[str, torch.Tensor],
                num_wrapping_dims: int = 0,
                **kwargs) -> torch.Tensor:
        embedder_keys = self._token_embedders.keys()
        input_keys = text_field_input.keys()

        # Check for unmatched keys
        if not self._allow_unmatched_keys:
            if embedder_keys < input_keys:
                # token embedder keys are a strict subset of text field input keys.
                message = (f"Your text field is generating more keys ({list(input_keys)}) "
                           f"than you have token embedders ({list(embedder_keys)}. "
                           f"If you are using a token embedder that requires multiple keys "
                           f"(for example, the OpenAI Transformer embedder or the BERT embedder) "
                           f"you need to add allow_unmatched_keys = True "
                           f"(and likely an embedder_to_indexer_map) to your "
                           f"BasicTextFieldEmbedder configuration. "
                           f"Otherwise, you should check that there is a 1:1 embedding "
                           f"between your token indexers and token embedders.")
                raise ConfigurationError(message)

            elif self._token_embedders.keys() != text_field_input.keys():
                # some other mismatch
                message = "Mismatched token keys: %s and %s" % (str(self._token_embedders.keys()),
                                                                str(text_field_input.keys()))
                raise ConfigurationError(message)

        embedded_representations = []
        keys = sorted(embedder_keys)
        for key in keys:
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, 'token_embedder_{}'.format(key))
            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]

            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            # If we pre-specified a mapping explictly, use that.
            # make mypy happy
            tensors: Union[List[Any], Dict[str, Any]] = None
            if self._embedder_to_indexer_map is not None:
                indexer_map = self._embedder_to_indexer_map[key]
                if isinstance(indexer_map, list):
                    # If `indexer_key` is None, we map it to `None`.
                    tensors = [(text_field_input[indexer_key] if indexer_key is not None else None)
                               for indexer_key in indexer_map]
                    token_vectors = embedder(*tensors, **forward_params_values)
                elif isinstance(indexer_map, dict):
                    tensors = {
                            name: text_field_input[argument]
                            for name, argument in indexer_map.items()
                    }
                    token_vectors = embedder(**tensors, **forward_params_values)
                else:
                    raise NotImplementedError
            else:
                # otherwise, we assume the mapping between indexers and embedders
                # is bijective and just use the key directly.
                tensors = [text_field_input[key]]
                token_vectors = embedder(*tensors, **forward_params_values)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

    # This is some unusual logic, it needs a custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BasicTextFieldEmbedder':  # type: ignore
        # pylint: disable=arguments-differ,bad-super-call

        # The original `from_params` for this class was designed in a way that didn't agree
        # with the constructor. The constructor wants a 'token_embedders' parameter that is a
        # `Dict[str, TokenEmbedder]`, but the original `from_params` implementation expected those
        # key-value pairs to be top-level in the params object.
        #
        # This breaks our 'configuration wizard' and configuration checks. Hence, going forward,
        # the params need a 'token_embedders' key so that they line up with what the constructor wants.
        # For now, the old behavior is still supported, but produces a DeprecationWarning.

        embedder_to_indexer_map = params.pop("embedder_to_indexer_map", None)
        if embedder_to_indexer_map is not None:
            embedder_to_indexer_map = embedder_to_indexer_map.as_dict(quiet=True)
        allow_unmatched_keys = params.pop_bool("allow_unmatched_keys", False)

        token_embedder_params = params.pop('token_embedders', None)

        if token_embedder_params is not None:
            # New way: explicitly specified, so use it.
            token_embedders = {
                    name: TokenEmbedder.from_params(subparams, vocab=vocab)
                    for name, subparams in token_embedder_params.items()
            }

        else:
            token_embedders = {}
            keys = list(params.keys())
            for key in keys:
                embedder_params = params.pop(key)
                token_embedders[key] = TokenEmbedder.from_params(vocab=vocab, params=embedder_params)

        params.assert_empty(cls.__name__)
        return cls(token_embedders, embedder_to_indexer_map, allow_unmatched_keys)
