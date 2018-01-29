
import torch
from overrides import overrides

from allennlp.common.params import Params
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

@SpanExtractor.register("locally_normalised")
class LocallyNormalisedSpanExtractor(SpanExtractor):
    """
    Represents spans as a function of the embeddings of their endpoints. Additionally,
    the width of the spans can be embedded and concatenated on to the final combination.

    Parameters
    ----------
    """
    def __init__(self,
                 input_dim: int,
                 max_span_width: int):
        super().__init__()
        self._global_attention = TimeDistributed(torch.nn.Linear(input_dim, 1))
        self._max_span_width = max_span_width

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                sequence_tensor: torch.FloatTensor,
                indicies: torch.LongTensor) -> None:
        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = indicies.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        span_widths = span_ends - span_starts

        # shape (batch_size, sequence_length, 1)
        global_attention_logits = self._global_attention(sequence_tensor)

        # Shape: (1, 1, max_span_width)
        max_span_range_indices = util.get_range_vector(self._max_span_width,
                                                       sequence_tensor.is_cuda).view(1, 1, -1)

        # Shape: (batch_size, num_spans, max_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # are of a smaller width than max_span_width, so we add this to the mask here.
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))

        # Shape: (batch_size, num_spans, max_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sequence_tensor, span_indices, flat_span_indices)

        # Shape: (batch_size, num_spans, max_span_width)
        span_attention_logits = util.batched_index_select(global_attention_logits,
                                                          span_indices,
                                                          flat_span_indices).squeeze(-1)
        # Shape: (batch_size, num_spans, max_span_width)
        span_attention_weights = util.last_dim_softmax(span_attention_logits, span_mask)

        # Do a weighted sum of the embedded spans with
        # respect to the normalised attention distributions.
        # Shape: (batch_size, num_spans, embedding_dim)
        attended_text_embeddings = util.weighted_sum(span_embeddings, span_attention_weights)

        return attended_text_embeddings

    @classmethod
    def from_params(cls, params: Params) -> "LocallyNormalisedSpanExtractor":
        input_dim = params.pop_int("input_dim")
        max_span_width = params.pop_int("max_span_width")
        return LocallyNormalisedSpanExtractor(input_dim=input_dim,
                                              max_span_width=max_span_width)
