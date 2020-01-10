# allennlp.modules.conditional_random_field

Conditional random field

## allowed_transitions
```python
allowed_transitions(constraint_type:str, labels:Dict[int, str]) -> List[Tuple[int, int]]
```

Given labels and a constraint type, returns the allowed transitions. It will
additionally include transitions for the start and end states, which are used
by the conditional random field.

Parameters
----------
constraint_type : ``str``, required
    Indicates which constraint to apply. Current choices are
    "BIO", "IOB1", "BIOUL", and "BMES".
labels : ``Dict[int, str]``, required
    A mapping {label_id -> label}. Most commonly this would be the value from
    Vocabulary.get_index_to_token_vocabulary()

Returns
-------
``List[Tuple[int, int]]``
    The allowed transitions (from_label_id, to_label_id).

## is_transition_allowed
```python
is_transition_allowed(constraint_type:str, from_tag:str, from_entity:str, to_tag:str, to_entity:str)
```

Given a constraint type and strings ``from_tag`` and ``to_tag`` that
represent the origin and destination of the transition, return whether
the transition is allowed under the given constraint type.

Parameters
----------
constraint_type : ``str``, required
    Indicates which constraint to apply. Current choices are
    "BIO", "IOB1", "BIOUL", and "BMES".
from_tag : ``str``, required
    The tag that the transition originates from. For example, if the
    label is ``I-PER``, the ``from_tag`` is ``I``.
from_entity : ``str``, required
    The entity corresponding to the ``from_tag``. For example, if the
    label is ``I-PER``, the ``from_entity`` is ``PER``.
to_tag : ``str``, required
    The tag that the transition leads to. For example, if the
    label is ``I-PER``, the ``to_tag`` is ``I``.
to_entity : ``str``, required
    The entity corresponding to the ``to_tag``. For example, if the
    label is ``I-PER``, the ``to_entity`` is ``PER``.

Returns
-------
``bool``
    Whether the transition is allowed under the given ``constraint_type``.

## ConditionalRandomField
```python
ConditionalRandomField(self, num_tags:int, constraints:List[Tuple[int, int]]=None, include_start_end_transitions:bool=True) -> None
```

This module uses the "forward-backward" algorithm to compute
the log-likelihood of its inputs assuming a conditional random field model.

See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

Parameters
----------
num_tags : ``int``, required
    The number of tags.
constraints : ``List[Tuple[int, int]]``, optional (default: None)
    An optional list of allowed transitions (from_tag_id, to_tag_id).
    These are applied to ``viterbi_tags()`` but do not affect ``forward()``.
    These should be derived from `allowed_transitions` so that the
    start and end transitions are handled correctly for your tag type.
include_start_end_transitions : ``bool``, optional (default: True)
    Whether to include the start and end transition parameters.

### forward
```python
ConditionalRandomField.forward(self, inputs:torch.Tensor, tags:torch.Tensor, mask:torch.ByteTensor=None) -> torch.Tensor
```

Computes the log likelihood.

### viterbi_tags
```python
ConditionalRandomField.viterbi_tags(self, logits:torch.Tensor, mask:torch.Tensor=None, top_k:int=None) -> Union[List[Tuple[List[int], float]], List[List[Tuple[List[int], float]]]]
```

Uses viterbi algorithm to find most likely tags for the given inputs.
If constraints are applied, disallows all other transitions.

Returns a list of results, of the same size as the batch (one result per batch member)
Each result is a List of length top_k, containing the top K viterbi decodings
Each decoding is a tuple  (tag_sequence, viterbi_score)

For backwards compatibility, if top_k is None, then instead returns a flat list of
tag sequences (the top tag sequence for each batch item).

