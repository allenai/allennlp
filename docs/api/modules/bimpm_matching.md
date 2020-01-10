# allennlp.modules.bimpm_matching

Multi-perspective matching layer

## multi_perspective_match
```python
multi_perspective_match(vector1:torch.Tensor, vector2:torch.Tensor, weight:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Calculate multi-perspective cosine matching between time-steps of vectors
of the same length.

Parameters
----------
vector1 : ``torch.Tensor``
    A tensor of shape ``(batch, seq_len, hidden_size)``
vector2 : ``torch.Tensor``
    A tensor of shape ``(batch, seq_len or 1, hidden_size)``
weight : ``torch.Tensor``
    A tensor of shape ``(num_perspectives, hidden_size)``

Returns
-------
A tuple of two tensors consisting multi-perspective matching results.
The first one is of the shape (batch, seq_len, 1), the second one is of shape
(batch, seq_len, num_perspectives)

## multi_perspective_match_pairwise
```python
multi_perspective_match_pairwise(vector1:torch.Tensor, vector2:torch.Tensor, weight:torch.Tensor, eps:float=1e-08) -> torch.Tensor
```

Calculate multi-perspective cosine matching between each time step of
one vector and each time step of another vector.

Parameters
----------
vector1 : ``torch.Tensor``
    A tensor of shape ``(batch, seq_len1, hidden_size)``
vector2 : ``torch.Tensor``
    A tensor of shape ``(batch, seq_len2, hidden_size)``
weight : ``torch.Tensor``
    A tensor of shape ``(num_perspectives, hidden_size)``
eps : ``float`` optional, (default = 1e-8)
    A small value to avoid zero division problem

Returns
-------
A tensor of shape (batch, seq_len1, seq_len2, num_perspectives) consisting
multi-perspective matching results

## BiMpmMatching
```python
BiMpmMatching(self, hidden_dim:int=100, num_perspectives:int=20, share_weights_between_directions:bool=True, is_forward:bool=None, with_full_match:bool=True, with_maxpool_match:bool=True, with_attentive_match:bool=True, with_max_attentive_match:bool=True) -> None
```

This ``Module`` implements the matching layer of BiMPM model described in `Bilateral
Multi-Perspective Matching for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_
by Zhiguo Wang et al., 2017.
Also please refer to the `TensorFlow implementation <https://github.com/zhiguowang/BiMPM/>`_ and
`PyTorch implementation <https://github.com/galsang/BIMPM-pytorch>`_.

Parameters
----------
hidden_dim : ``int``, optional (default = 100)
    The hidden dimension of the representations
num_perspectives : ``int``, optional (default = 20)
    The number of perspectives for matching
share_weights_between_directions : ``bool``, optional (default = True)
    If True, share weight between matching from sentence1 to sentence2 and from sentence2
    to sentence1, useful for non-symmetric tasks
is_forward : ``bool``, optional (default = None)
    Whether the matching is for forward sequence or backward sequence, useful in finding last
    token in full matching. It can not be None if with_full_match is True.
with_full_match : ``bool``, optional (default = True)
    If True, include full match
with_maxpool_match : ``bool``, optional (default = True)
    If True, include max pool match
with_attentive_match : ``bool``, optional (default = True)
    If True, include attentive match
with_max_attentive_match : ``bool``, optional (default = True)
    If True, include max attentive match

### forward
```python
BiMpmMatching.forward(self, context_1:torch.Tensor, mask_1:torch.Tensor, context_2:torch.Tensor, mask_2:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]
```

Given the forward (or backward) representations of sentence1 and sentence2, apply four bilateral
matching functions between them in one direction.

Parameters
----------
context_1 : ``torch.Tensor``
    Tensor of shape (batch_size, seq_len1, hidden_dim) representing the encoding of the first sentence.
mask_1 : ``torch.Tensor``
    Binary Tensor of shape (batch_size, seq_len1), indicating which
    positions in the first sentence are padding (0) and which are not (1).
context_2 : ``torch.Tensor``
    Tensor of shape (batch_size, seq_len2, hidden_dim) representing the encoding of the second sentence.
mask_2 : ``torch.Tensor``
    Binary Tensor of shape (batch_size, seq_len2), indicating which
    positions in the second sentence are padding (0) and which are not (1).

Returns
-------
A tuple of matching vectors for the two sentences. Each of which is a list of
matching vectors of shape (batch, seq_len, num_perspectives or 1)

