# allennlp.modules.pruner

## Pruner
```python
Pruner(self, scorer:torch.nn.modules.module.Module) -> None
```

This module scores and prunes items in a list using a parameterised scoring function and a
threshold.

Parameters
----------
scorer : ``torch.nn.Module``, required.
    A module which, given a tensor of shape (batch_size, num_items, embedding_size),
    produces a tensor of shape (batch_size, num_items, 1), representing a scalar score
    per item in the tensor.

### forward
```python
Pruner.forward(self, embeddings:torch.FloatTensor, mask:torch.LongTensor, num_items_to_keep:Union[int, torch.LongTensor]) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor]
```

Extracts the top-k scoring items with respect to the scorer. We additionally return
the indices of the top-k in their original order, not ordered by score, so that downstream
components can rely on the original ordering (e.g., for knowing what spans are valid
antecedents in a coreference resolution model). May use the same k for all sentences in
minibatch, or different k for each.

Parameters
----------
embeddings : ``torch.FloatTensor``, required.
    A tensor of shape (batch_size, num_items, embedding_size), containing an embedding for
    each item in the list that we want to prune.
mask : ``torch.LongTensor``, required.
    A tensor of shape (batch_size, num_items), denoting unpadded elements of
    ``embeddings``.
num_items_to_keep : ``Union[int, torch.LongTensor]``, required.
    If a tensor of shape (batch_size), specifies the number of items to keep for each
    individual sentence in minibatch.
    If an int, keep the same number of items for all sentences.

Returns
-------
top_embeddings : ``torch.FloatTensor``
    The representations of the top-k scoring items.
    Has shape (batch_size, max_num_items_to_keep, embedding_size).
top_mask : ``torch.LongTensor``
    The corresponding mask for ``top_embeddings``.
    Has shape (batch_size, max_num_items_to_keep).
top_indices : ``torch.IntTensor``
    The indices of the top-k scoring items into the original ``embeddings``
    tensor. This is returned because it can be useful to retain pointers to
    the original items, if each item is being scored by multiple distinct
    scorers, for instance. Has shape (batch_size, max_num_items_to_keep).
top_item_scores : ``torch.FloatTensor``
    The values of the top-k scoring items.
    Has shape (batch_size, max_num_items_to_keep, 1).

