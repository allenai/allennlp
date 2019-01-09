from typing import Tuple

from overrides import overrides
import torch

from allennlp.nn import util

class Pruner(torch.nn.Module):
    """
    This module scores and prunes items in a list using a parameterised scoring function and a
    threshold.

    Parameters
    ----------
    scorer : ``torch.nn.Module``, required.
        A module which, given a tensor of shape (batch_size, num_items, embedding_size),
        produces a tensor of shape (batch_size, num_items, 1), representing a scalar score
        per item in the tensor.
    """
    def __init__(self, scorer: torch.nn.Module) -> None:
        super().__init__()
        self._scorer = scorer

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                embeddings: torch.FloatTensor,
                mask: torch.LongTensor,
                num_items_to_keep: int) -> Tuple[torch.FloatTensor, torch.LongTensor,
                                                 torch.LongTensor, torch.FloatTensor]:
        """
        Extracts the top-k scoring items with respect to the scorer. We additionally return
        the indices of the top-k in their original order, not ordered by score, so that downstream
        components can rely on the original ordering (e.g., for knowing what spans are valid
        antecedents in a coreference resolution model).

        Parameters
        ----------
        embeddings : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_items, embedding_size), containing an embedding for
            each item in the list that we want to prune.
        mask : ``torch.LongTensor``, required.
            A tensor of shape (batch_size, num_items), denoting unpadded elements of
            ``embeddings``.
        num_items_to_keep : ``int``, required.
            The number of items to keep when pruning.

        Returns
        -------
        top_embeddings : ``torch.FloatTensor``
            The representations of the top-k scoring items.
            Has shape (batch_size, num_items_to_keep, embedding_size).
        top_mask : ``torch.LongTensor``
            The corresponding mask for ``top_embeddings``.
            Has shape (batch_size, num_items_to_keep).
        top_indices : ``torch.IntTensor``
            The indices of the top-k scoring items into the original ``embeddings``
            tensor. This is returned because it can be useful to retain pointers to
            the original items, if each item is being scored by multiple distinct
            scorers, for instance. Has shape (batch_size, num_items_to_keep).
        top_item_scores : ``torch.FloatTensor``
            The values of the top-k scoring items.
            Has shape (batch_size, num_items_to_keep, 1).
        """
        mask = mask.unsqueeze(-1)
        num_items = embeddings.size(1)
        # Shape: (batch_size, num_items, 1)
        scores = self._scorer(embeddings)

        if scores.size(-1) != 1 or scores.dim() != 3:
            raise ValueError(f"The scorer passed to Pruner must produce a tensor of shape"
                             f"(batch_size, num_items, 1), but found shape {scores.size()}")
        # Make sure that we don't select any masked items by setting their scores to be very
        # negative.  These are logits, typically, so -1e20 should be plenty negative.
        scores = util.replace_masked_values(scores, mask, -1e20)

        # Shape: (batch_size, num_items_to_keep, 1)
        _, top_indices = scores.topk(num_items_to_keep, 1)

        # Now we order the selected indices in increasing order with
        # respect to their indices (and hence, with respect to the
        # order they originally appeared in the ``embeddings`` tensor).
        top_indices, _ = torch.sort(top_indices, 1)

        # Shape: (batch_size, num_items_to_keep)
        top_indices = top_indices.squeeze(-1)

        # Shape: (batch_size * num_items_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select items for each element in the batch.
        flat_top_indices = util.flatten_and_batch_shift_indices(top_indices, num_items)

        # Shape: (batch_size, num_items_to_keep, embedding_size)
        top_embeddings = util.batched_index_select(embeddings, top_indices, flat_top_indices)
        # Shape: (batch_size, num_items_to_keep)
        top_mask = util.batched_index_select(mask, top_indices, flat_top_indices)

        # Shape: (batch_size, num_items_to_keep, 1)
        top_scores = util.batched_index_select(scores, top_indices, flat_top_indices)

        return top_embeddings, top_mask.squeeze(-1), top_indices, top_scores
