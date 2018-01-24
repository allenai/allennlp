from typing import Callable, Tuple

import torch

from allennlp.nn import util

class SpanPruner(torch.nn.Module):

    def __init__(self, scorer: Callable[[torch.FloatTensor], torch.FloatTensor]) -> None:
        super(SpanPruner, self).__init__()
        self._scorer = scorer


    def forward(self,
                span_embeddings: torch.FloatTensor,
                span_mask: torch.LongTensor,
                threshold: float,
                spans_to_keep: int) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        The indices of the top-k scoring spans according to span_scores. We return the
        indices in their original order, not ordered by score, so that we can rely on
        the ordering to consider the previous k spans as antecedents for each span later.
        Parameters
        ----------
        mention_scores : ``torch.FloatTensor``, required.
            The mention score for every candidate, with shape (batch_size, num_spans, 1).
        num_spans_to_keep : ``int``, required.
            The number of spans to keep when pruning.
        Returns
        -------
        top_span_indices : ``torch.IntTensor``, required.
            The indices of the top-k scoring spans. Has shape (batch_size, num_spans_to_keep).
        """
        num_spans = span_embeddings.size(1)
        # Shape: (batch_size, num_spans, 1)
        mention_scores = self._scorer(span_embeddings)
        mention_scores += span_mask.log()

        # Shape: (batch_size, num_spans_to_keep, 1)
        _, top_span_indices = mention_scores.topk(num_spans_to_keep, 1)
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

        return top_span_embeddings, top_span_mask, top_span_indices
