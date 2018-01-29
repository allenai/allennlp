from typing import Tuple

from overrides import overrides
import torch

from allennlp.nn import util

class SpanPruner(torch.nn.Module):
    """
    This module scores and prunes span-based representations using a parameterised scoring
    function and a threshold.

    Parameters
    ----------
    scorer : ``torch.nn.Module``, required.
        A module which, given a tensor of shape (batch_size, num_spans, embedding_size),
        produces a tensor of shape (batch_size, num_spans, 1), representing a scalar score
        per span in the tensor.
    """
    def __init__(self, scorer: torch.nn.Module) -> None:
        super(SpanPruner, self).__init__()
        self._scorer = scorer

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                span_embeddings: torch.FloatTensor,
                span_mask: torch.LongTensor,
                num_spans_to_keep: int) -> Tuple[torch.FloatTensor, torch.LongTensor,
                                                 torch.LongTensor, torch.FloatTensor]:
        """
        Extracts the top-k scoring spans with respect to the scorer. We additionally return
        the indices of the top-k in their original order, not ordered by score, so that we
        can rely on the ordering to consider the previous k spans as antecedents for each
        span later.

        Parameters
        ----------
        span_embeddings : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_spans, embedding_size), representing
            the set of embedded span representations.
        span_mask : ``torch.LongTensor``, required.
            A tensor of shape (batch_size, num_spans), denoting unpadded elements
            of ``span_embeddings``.
        num_spans_to_keep : ``int``, required.
            The number of spans to keep when pruning.

        Returns
        -------
        top_span_embeddings : ``torch.FloatTensor``
            The span representations of the top-k scoring spans.
            Has shape (batch_size, num_spans_to_keep, embedding_size).
        top_span_mask : ``torch.LongTensor``
            The coresponding mask for ``top_span_embeddings``.
            Has shape (batch_size, num_spans_to_keep).
        top_span_indices : ``torch.IntTensor``
            The indices of the top-k scoring spans into the original ``span_embeddings``
            tensor. This is returned because it can be useful to retain pointers to
            the original spans, if each span is being scored by multiple distinct
            scorers, for instance. Has shape (batch_size, num_spans_to_keep).
        top_span_scores : ``torch.FloatTensor``
            The values of the top-k scoring spans.
            Has shape (batch_size, num_spans_to_keep, 1).
        """
        span_mask = span_mask.unsqueeze(-1)
        num_spans = span_embeddings.size(1)
        # Shape: (batch_size, num_spans, 1)
        span_scores = self._scorer(span_embeddings)

        if span_scores.size(-1) != 1 or span_scores.dim() != 3:
            raise ValueError(f"The scorer passed to SpanPruner must produce a tensor of shape"
                             f"(batch_size, num_spans, 1), but found shape {span_scores.size()}")
        # Make sure that we don't select any masked spans by
        # setting their scores to be -inf.
        span_scores += span_mask.log()

        # Shape: (batch_size, num_spans_to_keep, 1)
        _, top_span_indices = span_scores.topk(num_spans_to_keep, 1)

        # Now we order the selected indices in increasing order with
        # respect to their indices (and hence, with respect to the
        # order they originally appeared in the ``span_embeddings`` tensor).
        top_span_indices, _ = torch.sort(top_span_indices, 1)

        # Shape: (batch_size, num_spans_to_keep)
        top_span_indices = top_span_indices.squeeze(-1)

        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Shape: (batch_size, num_spans_to_keep, embedding_size)
        top_span_embeddings = util.batched_index_select(span_embeddings,
                                                        top_span_indices,
                                                        flat_top_span_indices)
        # Shape: (batch_size, num_spans_to_keep)
        top_span_mask = util.batched_index_select(span_mask,
                                                  top_span_indices,
                                                  flat_top_span_indices)

        # Shape: (batch_size, num_spans_to_keep, 1)
        top_span_scores = util.batched_index_select(span_scores,
                                                    top_span_indices,
                                                    flat_top_span_indices)

        return top_span_embeddings, top_span_mask.squeeze(-1), top_span_indices, top_span_scores
