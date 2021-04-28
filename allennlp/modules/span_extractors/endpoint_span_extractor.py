import torch
from torch.nn.parameter import Parameter

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.span_extractors.span_extractor_with_span_width_embedding import (
    SpanExtractorWithSpanWidthEmbedding,
)
from allennlp.nn import util


@SpanExtractor.register("endpoint")
class EndpointSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Represents spans as a combination of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.

    The following types of representation are supported, assuming that
    `x = span_start_embeddings` and `y = span_end_embeddings`.

    `x`, `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations
    is performed elementwise.  You can list as many combinations as you want, comma separated.
    For example, you might give `x,y,x*y` as the `combination` parameter to this class.
    The computed similarity function would then be `[x; y; x*y]`, which can then be optionally
    concatenated with an embedded representation of the width of the span.

    Registered as a `SpanExtractor` with name "endpoint".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    combination : `str`, optional (default = `"x,y"`).
        The method used to combine the `start_embedding` and `end_embedding`
        representations. See above for a full description.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    use_exclusive_start_indices : `bool`, optional (default = `False`).
        If `True`, the start indices extracted are converted to exclusive indices. Sentinels
        are used to represent exclusive span indices for the elements in the first
        position in the sequence (as the exclusive indices for these elements are outside
        of the the sequence boundary) so that start indices can be exclusive.
        NOTE: This option can be helpful to avoid the pathological case in which you
        want span differences for length 1 spans - if you use inclusive indices, you
        will end up with an `x - x` operation for length 1 spans, which is not good.
    """

    def __init__(
        self,
        input_dim: int,
        combination: str = "x,y",
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
        use_exclusive_start_indices: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths,
        )
        self._combination = combination

        self._use_exclusive_start_indices = use_exclusive_start_indices
        if use_exclusive_start_indices:
            self._start_sentinel = Parameter(torch.randn([1, 1, int(input_dim)]))

    def get_output_dim(self) -> int:
        combined_dim = util.get_combined_dim(self._combination, [self._input_dim, self._input_dim])
        if self._span_width_embedding is not None:
            return combined_dim + self._span_width_embedding.get_output_dim()
        return combined_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> None:
        # shape (batch_size, num_spans)
        span_starts, span_ends = [index.squeeze(-1) for index in span_indices.split(1, dim=-1)]

        if span_indices_mask is not None:
            # It's not strictly necessary to multiply the span indices by the mask here,
            # but it's possible that the span representation was padded with something other
            # than 0 (such as -1, which would be an invalid index), so we do so anyway to
            # be safe.
            span_starts = span_starts * span_indices_mask
            span_ends = span_ends * span_indices_mask

        if not self._use_exclusive_start_indices:
            if sequence_tensor.size(-1) != self._input_dim:
                raise ValueError(
                    f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                    f"received ({self._input_dim})."
                )
            start_embeddings = util.batched_index_select(sequence_tensor, span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

        else:
            # We want `exclusive` span starts, so we remove 1 from the forward span starts
            # as the AllenNLP `SpanField` is inclusive.
            # shape (batch_size, num_spans)
            exclusive_span_starts = span_starts - 1
            # shape (batch_size, num_spans, 1)
            start_sentinel_mask = (exclusive_span_starts == -1).unsqueeze(-1)
            exclusive_span_starts = exclusive_span_starts * ~start_sentinel_mask.squeeze(-1)

            # We'll check the indices here at runtime, because it's difficult to debug
            # if this goes wrong and it's tricky to get right.
            if (exclusive_span_starts < 0).any():
                raise ValueError(
                    f"Adjusted span indices must lie inside the the sequence tensor, "
                    f"but found: exclusive_span_starts: {exclusive_span_starts}."
                )

            start_embeddings = util.batched_index_select(sequence_tensor, exclusive_span_starts)
            end_embeddings = util.batched_index_select(sequence_tensor, span_ends)

            # We're using sentinels, so we need to replace all the elements which were
            # outside the dimensions of the sequence_tensor with the start sentinel.
            start_embeddings = (
                start_embeddings * ~start_sentinel_mask + start_sentinel_mask * self._start_sentinel
            )

        combined_tensors = util.combine_tensors(
            self._combination, [start_embeddings, end_embeddings]
        )

        return combined_tensors
