
import torch
from overrides import overrides

from allennlp.common.params import Params
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn.util import batched_index_select, combine_tensors

@SpanExtractor.register("endpoint")
class EndpointSpanExtractor(SpanExtractor):
    """
    Represents spans as a function of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.

    The following types of representation are supported, assuming that
    ``x = span_start_embeddings`` and ``y = span_end_embeddings``.

    ``x``, ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations 
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give ``x,y,x*y`` as the ``combination`` parameter to this class.
    The computed similarity function would then be ``[x; y; x*y]``, which can then be optionally
    concatenated with an embedded representation of the width of the span.

    Parameters
    ----------
    combination : str, optional (default = "x-y").
        The method used to combine the ``start_embedding`` and ``end_embedding``
        representations. See above for a full description.
    span_width_embedding : ``Embedding``, optional (default = None).
        If passed, an embedded span width feature is concatenated onto
        the final span representation.
    """    
    def __init__(self, 
                 combination: str = "x-y", 
                 span_width_embedding: Embedding = None):
        super().__init__(self)
        self._combination = combination
        self._span_width_embedding = span_width_embedding

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                sequence_tensor: torch.FloatTensor,
                indicies: torch.LongTensor) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in indicies.split(2, dim=-1)]
        start_embeddings = batched_index_select(sequence_tensor, span_starts)
        end_embeddings = batched_index_select(sequence_tensor, span_ends)

        combined_tensors = combine_tensors(self._combination, [start_embeddings, end_embeddings])
        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(span_ends - span_starts)
            return torch.cat([combined_tensors, span_width_embeddings], -1)

        return combined_tensors

    @classmethod
    def from_params(cls, params: Params):
        combination = params.pop("combination", "x-y")
        span_width_embedding_params = params.pop("span_width_embedding", None)

        if span_width_embedding_params:
            span_width_embedding = Embedding.from_params(span_width_embedding_params)
        else:
            span_width_embedding = None

        return EndpointSpanExtractor(combination=combination,
                                     span_width_embedding=span_width_embedding)
