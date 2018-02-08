
import torch
from overrides import overrides

from allennlp.common.params import Params
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn.util import batched_index_select, combine_tensors, bucket_values, get_combined_dim, add_sentence_boundary_token_ids
from allennlp.common.checks import ConfigurationError


@SpanExtractor.register("bidirectional_endpoint")
class BidirectionalEndpointSpanExtractor(SpanExtractor):
    """
    Represents spans from a bidirectional encoder as a concatenation of two different
    span representations of the span endpoints, one for the forward direction of the encoder
    and one from the backward direction. This type of representation encodes some subtelty,
    because when you consider the forward and backward directions separately, the end index
    of the span for the backward direction's representation is actually the start index.

    By default, this ``SpanExtractor`` represents spans as 
    (inclusive end index) - (exclusive start index), meaning that the representation is
    the difference between the the last word in the span and the word `before` the span
    started. Note that this is direction agnostic, so for the forward direction this will
    be (end_index - start_index), but for the backward direction, this will be 
    (start_index - end_index), as in the backward direction, the 'first' index is actually
    the end_index.

    Additionally, the width of the spans can be embedded and concatenated on to the
    final combination.

    The following other types of representation are supported for both the forward and backward 
    directions, assuming that ``x = span_start_embeddings`` and ``y = span_end_embeddings``.

    ``x``, ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give ``x,y,x*y`` as the ``combination`` parameter to this class.
    The computed similarity function would then be ``[x; y; x*y]``, which can then be optionally
    concatenated with an embedded representation of the width of the span.

    Parameters
    ----------
    input_dim : ``int``, required.
        The final dimension of the ``sequence_tensor``.
    forward_combination : str, optional (default = "y-x").
        The method used to combine the ``forward_start_embeddings`` and ``forward_end_embeddings``
        for the forward direction of the bidirectional representation.
        See above for a full description.
    backward_combination : str, optional (default = "x-y").
        The method used to combine the ``backward_start_embeddings`` and ``backward_end_embeddings``
        for the backward direction of the bidirectional representation.
        See above for a full description.
    num_width_embeddings : ``int``, optional (default = None).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : ``int``, optional (default = None).
        The embedding size for the span_width features.
    bucket_widths : ``bool``, optional (default = False).
        Whether to bucket the span widths into log-space buckets. If ``False``,
        the raw span widths are used.
    """
    def __init__(self,
                 input_dim: int,
                 forward_combination: str = "y-x",
                 backward_combination: str = "x-y",
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._forward_combination = forward_combination
        self._backward_combination = backward_combination
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        if self._input_dim % 2 != 0:
            raise ConfigurationError("The input dimension is not divisible by 2, but the "
                                     "BidirectionalEndpointSpanExtractor assumes the embedded representation "
                                     "is bidirectional (and hence divisible by 2).")
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(num_width_embeddings, span_width_embedding_dim)
        elif not all([num_width_embeddings is None, span_width_embedding_dim is None]):
            raise ConfigurationError("To use a span width embedding representation, you must"
                                     "specify both num_width_buckets and span_width_embedding_dim.")
        else:
            self._span_width_embedding = None

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        combined_dim = get_combined_dim(self._combination, [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                sequence_tensor: torch.FloatTensor,
                indicies: torch.LongTensor) -> None:

        # Both of shape (batch_size, sequence_length, embedding_size / 2)
        forward_sequence, backward_sequence = torch.split(sequence_tensor, 2, -1)

        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in indicies.split(1, dim=-1)]

        # We want `exclusive` span starts, so we remove 1 from the forward span starts
        # as the AllenNLP ``SpanField`` is inclusive.
        exclusive_span_starts = span_starts - 1
        start_sentinel_mask = (exclusive_span_starts == -1).long()

        # We want `exclusive` span ends, so we remove 1 from the forward span starts
        # as the AllenNLP ``SpanField`` is inclusive.
        exclusive_span_ends = span_ends + 1
        end_sentinel_mask = (exclusive_span_ends == sequence_lengths + 1)

        forward_start_embeddings = batched_index_select(forward_sequence, exclusive_span_starts)
        forward_end_embeddings = batched_index_select(forward_sequence, span_ends)

        backward_start_embeddings = batched_index_select(backward_sequence, span_starts)
        backward_end_embeddings = batched_index_select(backward_sequence, exclusive_span_ends)

        forward_spans = combine_tensors(self._forward_combination, [forward_start_embeddings, forward_end_embeddings])
        backward_spans = combine_tensors(self._combination, [backward_start_embeddings, backward_end_embeddings])
        span_embeddings = torch.cat([forward_spans, backward_spans], -1)

        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = bucket_values(span_ends - span_starts,
                                            num_total_buckets=self._num_width_embeddings)
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            return torch.cat([span_embeddings, span_width_embeddings], -1)
        return span_embeddings

    @classmethod
    def from_params(cls, params: Params) -> "BidirectionalEndpointSpanExtractor":
        input_dim = params.pop_int("input_dim")
        forward_combination = params.pop("forward_combination", "y-x")
        backward_combination = params.pop("backward_combination", "x-y")
        num_width_embeddings = params.pop_int("num_width_embeddings", None)
        span_width_embedding_dim = params.pop_int("span_width_embedding_dim", None)
        bucket_widths = params.pop_bool("bucket_widths", False)
        return BidirectionalEndpointSpanExtractor(input_dim=input_dim,
                                     forward_combination=forward_combination,
                                     backward_combination=backward_combination,
                                     num_width_embeddings=num_width_embeddings,
                                     span_width_embedding_dim=span_width_embedding_dim,
                                     bucket_widths=bucket_widths)
