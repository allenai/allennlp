
import torch
from torch.nn.parameter import Parameter
from overrides import overrides

from allennlp.common.params import Params
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import util
from allennlp.common.checks import ConfigurationError


@SpanExtractor.register("bidirectional_endpoint")
class BidirectionalEndpointSpanExtractor(SpanExtractor):
    """
    Represents spans from a bidirectional encoder as a concatenation of two different
    representations of the span endpoints, one for the forward direction of the encoder
    and one from the backward direction. This type of representation encodes some subtelty,
    because when you consider the forward and backward directions separately, the end index
    of the span for the backward direction's representation is actually the start index.

    By default, this ``SpanExtractor`` represents spans as
    ``sequence_tensor[inclusive_span_end] - sequence_tensor[exclusive_span_start]``
    meaning that the representation is the difference between the the last word in the span
    and the word `before` the span started. Note that the start and end indices are with
    respect to the direction that the RNN is going in, so for the backward direction, the
    start/end indices are reversed.

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
    backward_combination : str, optional (default = "y-x").
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
    use_sentinels : ``bool``, optional (default = ``True``).
        If ``True``, sentinels are used to represent exclusive span indices for the elements
        in the first and last positions in the sequence (as the exclusive indices for these
        elements are outside of the the sequence boundary). This is not strictly necessary,
        as you may know that your exclusive start and end indices are always within your sequence
        representation, such as if you have appended/prepended <START> and <END> tokens to your
        sequence.
    """
    def __init__(self,
                 input_dim: int,
                 forward_combination: str = "y-x",
                 backward_combination: str = "y-x",
                 num_width_embeddings: int = None,
                 span_width_embedding_dim: int = None,
                 bucket_widths: bool = False,
                 use_sentinels: bool = True) -> None:
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

        self._use_sentinels = use_sentinels
        if use_sentinels:
            self._start_sentinel = Parameter(torch.randn([1, 1, int(input_dim / 2)]))
            self._end_sentinel = Parameter(torch.randn([1, 1, int(input_dim / 2)]))

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        unidirectional_dim = int(self._input_dim / 2)
        forward_combined_dim = util.get_combined_dim(self._forward_combination,
                                                     [unidirectional_dim, unidirectional_dim])
        backward_combined_dim = util.get_combined_dim(self._backward_combination,
                                                      [unidirectional_dim, unidirectional_dim])
        if self._span_width_embedding is not None:
            return forward_combined_dim + backward_combined_dim + \
                   self._span_width_embedding.get_output_dim()
        return forward_combined_dim + backward_combined_dim

    @overrides
    def forward(self,
                sequence_tensor: torch.FloatTensor,
                span_indices: torch.LongTensor,
                sequence_mask: torch.LongTensor = None,
                span_indices_mask: torch.LongTensor = None) -> torch.FloatTensor:

        # Both of shape (batch_size, sequence_length, embedding_size / 2)
        forward_sequence, backward_sequence = sequence_tensor.split(int(self._input_dim / 2), dim=-1)
        forward_sequence = forward_sequence.contiguous()
        backward_sequence = backward_sequence.contiguous()

        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

        if span_indices_mask is not None:
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask
        # We want `exclusive` span starts, so we remove 1 from the forward span starts
        # as the AllenNLP ``SpanField`` is inclusive.
        # shape (batch_size, num_spans)
        exclusive_span_starts = span_starts - 1
        # shape (batch_size, num_spans, 1)
        start_sentinel_mask = (exclusive_span_starts == -1).long().unsqueeze(-1)

        # We want `exclusive` span ends for the backward direction
        # (so that the `start` of the span in that direction is exlusive), so
        # we add 1 to the span ends as the AllenNLP ``SpanField`` is inclusive.
        exclusive_span_ends = span_ends + 1

        if sequence_mask is not None:
            # shape (batch_size)
            sequence_lengths = util.get_lengths_from_binary_sequence_mask(sequence_mask)
        else:
            # shape (batch_size), filled with the sequence length size of the sequence_tensor.
            sequence_lengths = (torch.ones_like(sequence_tensor[:, 0, 0], dtype=torch.long) *
                                sequence_tensor.size(1))

        # shape (batch_size, num_spans, 1)
        end_sentinel_mask = (exclusive_span_ends == sequence_lengths.unsqueeze(-1)).long().unsqueeze(-1)

        # As we added 1 to the span_ends to make them exclusive, which might have caused indices
        # equal to the sequence_length to become out of bounds, we multiply by the inverse of the
        # end_sentinel mask to erase these indices (as we will replace them anyway in the block below).
        # The same argument follows for the exclusive span start indices.
        exclusive_span_ends = exclusive_span_ends * (1 - end_sentinel_mask.squeeze(-1))
        exclusive_span_starts = exclusive_span_starts * (1 - start_sentinel_mask.squeeze(-1))

        # We'll check the indices here at runtime, because it's difficult to debug
        # if this goes wrong and it's tricky to get right.
        if (exclusive_span_starts < 0).any() or (exclusive_span_ends > sequence_lengths.unsqueeze(-1)).any():
            raise ValueError(f"Adjusted span indices must lie inside the length of the sequence tensor, "
                             f"but found: exclusive_span_starts: {exclusive_span_starts}, "
                             f"exclusive_span_ends: {exclusive_span_ends} for a sequence tensor with lengths "
                             f"{sequence_lengths}.")

        # Forward Direction: start indices are exclusive. Shape (batch_size, num_spans, input_size / 2)
        forward_start_embeddings = util.batched_index_select(forward_sequence, exclusive_span_starts)
        # Forward Direction: end indices are inclusive, so we can just use span_ends.
        # Shape (batch_size, num_spans, input_size / 2)
        forward_end_embeddings = util.batched_index_select(forward_sequence, span_ends)

        # Backward Direction: The backward start embeddings use the `forward` end
        # indices, because we are going backwards.
        # Shape (batch_size, num_spans, input_size / 2)
        backward_start_embeddings = util.batched_index_select(backward_sequence, exclusive_span_ends)
        # Backward Direction: The backward end embeddings use the `forward` start
        # indices, because we are going backwards.
        # Shape (batch_size, num_spans, input_size / 2)
        backward_end_embeddings = util.batched_index_select(backward_sequence, span_starts)

        if self._use_sentinels:
            # If we're using sentinels, we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with either the start sentinel,
            # or the end sentinel.
            float_end_sentinel_mask = end_sentinel_mask.float()
            float_start_sentinel_mask = start_sentinel_mask.float()
            forward_start_embeddings = forward_start_embeddings * (1 - float_start_sentinel_mask) \
                                        + float_start_sentinel_mask * self._start_sentinel
            backward_start_embeddings = backward_start_embeddings * (1 - float_end_sentinel_mask) \
                                        + float_end_sentinel_mask * self._end_sentinel

        # Now we combine the forward and backward spans in the manner specified by the
        # respective combinations and concatenate these representations.
        # Shape (batch_size, num_spans, forward_combination_dim)
        forward_spans = util.combine_tensors(self._forward_combination,
                                             [forward_start_embeddings, forward_end_embeddings])
        # Shape (batch_size, num_spans, backward_combination_dim)
        backward_spans = util.combine_tensors(self._backward_combination,
                                              [backward_start_embeddings, backward_end_embeddings])
        # Shape (batch_size, num_spans, forward_combination_dim + backward_combination_dim)
        span_embeddings = torch.cat([forward_spans, backward_spans], -1)

        if self._span_width_embedding is not None:
            # Embed the span widths and concatenate to the rest of the representations.
            if self._bucket_widths:
                span_widths = util.bucket_values(span_ends - span_starts,
                                                 num_total_buckets=self._num_width_embeddings)
            else:
                span_widths = span_ends - span_starts

            span_width_embeddings = self._span_width_embedding(span_widths)
            return torch.cat([span_embeddings, span_width_embeddings], -1)

        if span_indices_mask is not None:
            return span_embeddings * span_indices_mask.float().unsqueeze(-1)
        return span_embeddings

    @classmethod
    def from_params(cls, params: Params) -> "BidirectionalEndpointSpanExtractor":
        input_dim = params.pop_int("input_dim")
        forward_combination = params.pop("forward_combination", "y-x")
        backward_combination = params.pop("backward_combination", "x-y")
        num_width_embeddings = params.pop_int("num_width_embeddings", None)
        span_width_embedding_dim = params.pop_int("span_width_embedding_dim", None)
        bucket_widths = params.pop_bool("bucket_widths", False)
        use_sentinels = params.pop_bool("use_sentinels", True)
        return BidirectionalEndpointSpanExtractor(input_dim=input_dim,
                                                  forward_combination=forward_combination,
                                                  backward_combination=backward_combination,
                                                  num_width_embeddings=num_width_embeddings,
                                                  span_width_embedding_dim=span_width_embedding_dim,
                                                  bucket_widths=bucket_widths,
                                                  use_sentinels=use_sentinels)
