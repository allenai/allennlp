# allennlp.modules.token_embedders.elmo_token_embedder_multilang

## ElmoTokenEmbedderMultiLang
```python
ElmoTokenEmbedderMultiLang(self, options_files:Dict[str, str], weight_files:Dict[str, str], do_layer_norm:bool=False, dropout:float=0.5, requires_grad:bool=False, projection_dim:int=None, vocab_to_cache:List[str]=None, scalar_mix_parameters:List[float]=None, aligning_files:Dict[str, str]=None) -> None
```

A multilingual ELMo embedder - extending ElmoTokenEmbedder for multiple languages.
Each language has different weights for the ELMo model and an alignment matrix.

Parameters
----------
options_files : ``Dict[str, str]``, required.
    A dictionary of language identifier to an ELMo JSON options file.
weight_files : ``Dict[str, str]``, required.
    A dictionary of language identifier to an ELMo hdf5 weight file.
do_layer_norm : ``bool``, optional.
    Should we apply layer normalization (passed to ``ScalarMix``)?
dropout : ``float``, optional.
    The dropout value to be applied to the ELMo representations.
requires_grad : ``bool``, optional
    If True, compute gradient of ELMo parameters for fine tuning.
projection_dim : ``int``, optional
    If given, we will project the ELMo embedding down to this dimension.  We recommend that you
    try using ELMo with a lot of dropout and no projection first, but we have found a few cases
    where projection helps (particulary where there is very limited training data).
vocab_to_cache : ``List[str]``, optional, (default = 0.5).
    A list of words to pre-compute and cache character convolutions
    for. If you use this option, the ElmoTokenEmbedder expects that you pass word
    indices of shape (batch_size, timesteps) to forward, instead
    of character indices. If you use this option and pass a word which
    wasn't pre-cached, this will break.
scalar_mix_parameters : ``List[int]``, optional, (default=None).
    If not ``None``, use these scalar mix parameters to weight the representations
    produced by different layers. These mixing weights are not updated during
    training.
aligning_files : ``Dict[str, str]``, optional, (default={}).
    A dictionary of language identifier to a pth file with an alignment matrix.

### forward
```python
ElmoTokenEmbedderMultiLang.forward(self, inputs:torch.Tensor, lang:str, word_inputs:torch.Tensor=None) -> torch.Tensor
```

Parameters
----------
inputs : ``torch.Tensor``
    Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
lang : ``str``, , required.
    The language of the ELMo embedder to use.
word_inputs : ``torch.Tensor``, optional.
    If you passed a cached vocab, you can in addition pass a tensor of shape
    ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.

Returns
-------
The ELMo representations for the given language for the input sequence, shape
``(batch_size, timesteps, embedding_dim)``

