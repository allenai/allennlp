# allennlp.modules.elmo

## Elmo
```python
Elmo(self, options_file:str, weight_file:str, num_output_representations:int, requires_grad:bool=False, do_layer_norm:bool=False, dropout:float=0.5, vocab_to_cache:List[str]=None, keep_sentence_boundaries:bool=False, scalar_mix_parameters:List[float]=None, module:torch.nn.modules.module.Module=None) -> None
```

Compute ELMo representations using a pre-trained bidirectional language model.

See "Deep contextualized word representations", Peters et al. for details.

This module takes character id input and computes ``num_output_representations`` different layers
of ELMo representations.  Typically ``num_output_representations`` is 1 or 2.  For example, in
the case of the SRL model in the above paper, ``num_output_representations=1`` where ELMo was included at
the input token representation layer.  In the case of the SQuAD model, ``num_output_representations=2``
as ELMo was also included at the GRU output layer.

In the implementation below, we learn separate scalar weights for each output layer,
but only run the biLM once on each input sequence for efficiency.

Parameters
----------
options_file : ``str``, required.
    ELMo JSON options file
weight_file : ``str``, required.
    ELMo hdf5 weight file
num_output_representations : ``int``, required.
    The number of ELMo representation to output with
    different linear weighted combination of the 3 layers (i.e.,
    character-convnet output, 1st lstm output, 2nd lstm output).
requires_grad : ``bool``, optional
    If True, compute gradient of ELMo parameters for fine tuning.
do_layer_norm : ``bool``, optional, (default = False).
    Should we apply layer normalization (passed to ``ScalarMix``)?
dropout : ``float``, optional, (default = 0.5).
    The dropout to be applied to the ELMo representations.
vocab_to_cache : ``List[str]``, optional, (default = None).
    A list of words to pre-compute and cache character convolutions
    for. If you use this option, Elmo expects that you pass word
    indices of shape (batch_size, timesteps) to forward, instead
    of character indices. If you use this option and pass a word which
    wasn't pre-cached, this will break.
keep_sentence_boundaries : ``bool``, optional, (default = False)
    If True, the representation of the sentence boundary tokens are
    not removed.
scalar_mix_parameters : ``List[float]``, optional, (default = None)
    If not ``None``, use these scalar mix parameters to weight the representations
    produced by different layers. These mixing weights are not updated during
    training. The mixing weights here should be the unnormalized (i.e., pre-softmax)
    weights. So, if you wanted to use only the 1st layer of a 2-layer ELMo,
    you can set this to [-9e10, 1, -9e10 ].
module : ``torch.nn.Module``, optional, (default = None).
    If provided, then use this module instead of the pre-trained ELMo biLM.
    If using this option, then pass ``None`` for both ``options_file``
    and ``weight_file``.  The module must provide a public attribute
    ``num_layers`` with the number of internal layers and its ``forward``
    method must return a ``dict`` with ``activations`` and ``mask`` keys
    (see `_ElmoBilm`` for an example).  Note that ``requires_grad`` is also
    ignored with this option.

### forward
```python
Elmo.forward(self, inputs:torch.Tensor, word_inputs:torch.Tensor=None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
```

Parameters
----------
inputs : ``torch.Tensor``, required.
Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
word_inputs : ``torch.Tensor``, required.
    If you passed a cached vocab, you can in addition pass a tensor of shape
    ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.

Returns
-------
Dict with keys:
``'elmo_representations'`` : ``List[torch.Tensor]``
    A ``num_output_representations`` list of ELMo representations for the input sequence.
    Each representation is shape ``(batch_size, timesteps, embedding_dim)``
``'mask'``:  ``torch.Tensor``
    Shape ``(batch_size, timesteps)`` long tensor with sequence mask.

## batch_to_ids
```python
batch_to_ids(batch:List[List[str]]) -> torch.Tensor
```

Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
(len(batch), max sentence length, max word length).

Parameters
----------
batch : ``List[List[str]]``, required
    A list of tokenized sentences.

Returns
-------
    A tensor of padded character ids.

