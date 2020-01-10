# allennlp.nn.beam_search

## BeamSearch
```python
BeamSearch(self, end_index:int, max_steps:int=50, beam_size:int=10, per_node_beam_size:int=None) -> None
```

Implements the beam search algorithm for decoding the most likely sequences.

Parameters
----------
end_index : ``int``
    The index of the "stop" or "end" token in the target vocabulary.
max_steps : ``int``, optional (default = 50)
    The maximum number of decoding steps to take, i.e. the maximum length
    of the predicted sequences.
beam_size : ``int``, optional (default = 10)
    The width of the beam used.
per_node_beam_size : ``int``, optional (default = beam_size)
    The maximum number of candidates to consider per node, at each step in the search.
    If not given, this just defaults to ``beam_size``. Setting this parameter
    to a number smaller than ``beam_size`` may give better results, as it can introduce
    more diversity into the search. See `Beam Search Strategies for Neural Machine Translation.
    Freitag and Al-Onaizan, 2017 <https://arxiv.org/abs/1702.01806>`_.

### search
```python
BeamSearch.search(self, start_predictions:torch.Tensor, start_state:Dict[str, torch.Tensor], step:Callable[[torch.Tensor, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor]
```

Given a starting state and a step function, apply beam search to find the
most likely target sequences.

Notes
-----
If your step function returns ``-inf`` for some log probabilities
(like if you're using a masked log-softmax) then some of the "best"
sequences returned may also have ``-inf`` log probability. Specifically
this happens when the beam size is smaller than the number of actions
with finite log probability (non-zero probability) returned by the step function.
Therefore if you're using a mask you may want to check the results from ``search``
and potentially discard sequences with non-finite log probability.

Parameters
----------
start_predictions : ``torch.Tensor``
    A tensor containing the initial predictions with shape ``(batch_size,)``.
    Usually the initial predictions are just the index of the "start" token
    in the target vocabulary.
start_state : ``StateType``
    The initial state passed to the ``step`` function. Each value of the state dict
    should be a tensor of shape ``(batch_size, *)``, where ``*`` means any other
    number of dimensions.
step : ``StepFunctionType``
    A function that is responsible for computing the next most likely tokens,
    given the current state and the predictions from the last time step.
    The function should accept two arguments. The first being a tensor
    of shape ``(group_size,)``, representing the index of the predicted
    tokens from the last time step, and the second being the current state.
    The ``group_size`` will be ``batch_size * beam_size``, except in the initial
    step, for which it will just be ``batch_size``.
    The function is expected to return a tuple, where the first element
    is a tensor of shape ``(group_size, target_vocab_size)`` containing
    the log probabilities of the tokens for the next step, and the second
    element is the updated state. The tensor in the state should have shape
    ``(group_size, *)``, where ``*`` means any other number of dimensions.

Returns
-------
Tuple[torch.Tensor, torch.Tensor]
    Tuple of ``(predictions, log_probabilities)``, where ``predictions``
    has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
    has shape ``(batch_size, beam_size)``.

