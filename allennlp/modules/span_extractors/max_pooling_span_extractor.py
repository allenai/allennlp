import torch

from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.span_extractors.span_extractor_with_span_width_embedding import (
    SpanExtractorWithSpanWidthEmbedding,
)
from allennlp.nn import util
from allennlp.nn.util import masked_max


@SpanExtractor.register("max_pooling")
class MaxPoolingSpanExtractor(SpanExtractorWithSpanWidthEmbedding):
    """
    Represents spans through the application of a dimension-wise max-pooling operation.
    Given a span x_i, ..., x_j with i,j as span_start and span_end, each dimension d
    of the resulting span s is computed via s_d = max(x_id, ..., x_jd).

    Elements masked-out by sequence_mask are ignored when max-pooling is computed.
    Span representations of masked out span_indices by span_mask are set to '0.'

    Registered as a `SpanExtractor` with name "max_pooling".

    # Parameters

    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.

    # Returns

    max_pooling_text_embeddings : `torch.FloatTensor`.
        A tensor of shape (batch_size, num_spans, input_dim), which each span representation
        is the result of a max-pooling operation.

    """

    def __init__(
        self,
        input_dim: int,
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_width_embeddings=num_width_embeddings,
            span_width_embedding_dim=span_width_embedding_dim,
            bucket_widths=bucket_widths,
        )

    def get_output_dim(self) -> int:
        if self._span_width_embedding is not None:
            return self._input_dim + self._span_width_embedding.get_output_dim()
        return self._input_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> torch.FloatTensor:

        if sequence_tensor.size(-1) != self._input_dim:
            raise ValueError(
                f"Dimension mismatch expected ({sequence_tensor.size(-1)}) "
                f"received ({self._input_dim})."
            )

        if sequence_tensor.shape[1] <= span_indices.max() or span_indices.min() < 0:
            raise IndexError(
                f"Span index out of range, max index ({span_indices.max()}) "
                f"or min index ({span_indices.min()}) "
                f"not valid for sequence of length ({sequence_tensor.shape[1]})."
            )

        if (span_indices[:, :, 0] > span_indices[:, :, 1]).any():
            raise IndexError(
                "Span start above span end",
            )

        # Calculate the maximum sequence length for each element in batch.
        # If span_end indices are above these length, we adjust the indices in adapted_span_indices
        if sequence_mask is not None:
            # shape (batch_size)
            sequence_lengths = util.get_lengths_from_binary_sequence_mask(sequence_mask)
        else:
            # shape (batch_size), filled with the sequence length size of the sequence_tensor.
            sequence_lengths = torch.ones_like(
                sequence_tensor[:, 0, 0], dtype=torch.long
            ) * sequence_tensor.size(1)

        adapted_span_indices = torch.tensor(span_indices, device=span_indices.device)

        for b in range(sequence_lengths.shape[0]):
            adapted_span_indices[b, :, 1][adapted_span_indices[b, :, 1] >= sequence_lengths[b]] = (
                sequence_lengths[b] - 1
            )

        # Raise Error if span indices were completely masked by sequence mask.
        # We only adjust span_end to the last valid index, so if span_end is below span_start,
        # both were above the max index:

        if (adapted_span_indices[:, :, 0] > adapted_span_indices[:, :, 1]).any():
            raise IndexError(
                "Span indices were masked out entirely by sequence mask",
            )

        # span_vals <- (batch x num_spans x max_span_length x dim)
        span_vals, span_mask = util.batched_span_select(sequence_tensor, adapted_span_indices)

        # The application of masked_max requires a mask of the same shape as span_vals
        # We repeat the mask along the last dimension (embedding dimension)
        repeat_dim = len(span_vals.shape) - 1
        repeat_idx = [1] * (repeat_dim) + [span_vals.shape[-1]]

        # ext_span_mask <- (batch x num_spans x max_span_length x dim)
        # ext_span_mask True for values in span, False for masked out values
        ext_span_mask = span_mask.unsqueeze(repeat_dim).repeat(repeat_idx)

        # max_out <- (batch x num_spans x dim)
        max_out = masked_max(span_vals, ext_span_mask, dim=-2)

        return max_out
