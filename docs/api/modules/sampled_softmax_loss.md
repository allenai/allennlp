# allennlp.modules.sampled_softmax_loss

## SampledSoftmaxLoss
```python
SampledSoftmaxLoss(self, num_words:int, embedding_dim:int, num_samples:int, sparse:bool=False, unk_id:int=None, use_character_inputs:bool=True, use_fast_sampler:bool=False) -> None
```

Based on the default log_uniform_candidate_sampler in tensorflow.

NOTE: num_words DOES NOT include padding id.

NOTE: In all cases except (tie_embeddings=True and use_character_inputs=False)
the weights are dimensioned as num_words and do not include an entry for the padding (0) id.
For the (tie_embeddings=True and use_character_inputs=False) case,
then the embeddings DO include the extra 0 padding, to be consistent with the word embedding layer.

Parameters
----------
num_words, ``int``, required
    The number of words in the vocabulary
embedding_dim, ``int``, required
    The dimension to softmax over
num_samples, ``int``, required
    During training take this many samples. Must be less than num_words.
sparse, ``bool``, optional (default = False)
    If this is true, we use a sparse embedding matrix.
unk_id, ``int``, optional (default = None)
    If provided, the id that represents unknown characters.
use_character_inputs, ``bool``, optional (default = True)
    Whether to use character inputs
use_fast_sampler, ``bool``, optional (default = False)
    Whether to use the fast cython sampler.

