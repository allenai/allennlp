# allennlp.models.encoder_decoders.copynet_seq2seq

## CopyNetSeq2Seq
```python
CopyNetSeq2Seq(self, vocab:allennlp.data.vocabulary.Vocabulary, source_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, attention:allennlp.modules.attention.attention.Attention, beam_size:int, max_decoding_steps:int, target_embedding_dim:int=30, copy_token:str='@COPY@', source_namespace:str='source_tokens', target_namespace:str='target_tokens', tensor_based_metric:allennlp.training.metrics.metric.Metric=None, token_based_metric:allennlp.training.metrics.metric.Metric=None, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x139a473c8>) -> None
```

This is an implementation of `CopyNet <https://arxiv.org/pdf/1603.06393>`_.
CopyNet is a sequence-to-sequence encoder-decoder model with a copying mechanism
that can copy tokens from the source sentence into the target sentence instead of
generating all target tokens only from the target vocabulary.

It is very similar to a typical seq2seq model used in neural machine translation
tasks, for example, except that in addition to providing a "generation" score at each timestep
for the tokens in the target vocabulary, it also provides a "copy" score for each
token that appears in the source sentence. In other words, you can think of CopyNet
as a seq2seq model with a dynamic target vocabulary that changes based on the tokens
in the source sentence, allowing it to predict tokens that are out-of-vocabulary (OOV)
with respect to the actual target vocab.

Parameters
----------
vocab : ``Vocabulary``, required
    Vocabulary containing source and target vocabularies.
source_embedder : ``TextFieldEmbedder``, required
    Embedder for source side sequences
encoder : ``Seq2SeqEncoder``, required
    The encoder of the "encoder/decoder" model
attention : ``Attention``, required
    This is used to get a dynamic summary of encoder outputs at each timestep
    when producing the "generation" scores for the target vocab.
beam_size : ``int``, required
    Beam width to use for beam search prediction.
max_decoding_steps : ``int``, required
    Maximum sequence length of target predictions.
target_embedding_dim : ``int``, optional (default = 30)
    The size of the embeddings for the target vocabulary.
copy_token : ``str``, optional (default = '@COPY@')
    The token used to indicate that a target token was copied from the source.
    If this token is not already in your target vocabulary, it will be added.
source_namespace : ``str``, optional (default = 'source_tokens')
    The namespace for the source vocabulary.
target_namespace : ``str``, optional (default = 'target_tokens')
    The namespace for the target vocabulary.
tensor_based_metric : ``Metric``, optional (default = BLEU)
    A metric to track on validation data that takes raw tensors when its called.
    This metric must accept two arguments when called: a batched tensor
    of predicted token indices, and a batched tensor of gold token indices.
token_based_metric : ``Metric``, optional (default = None)
    A metric to track on validation data that takes lists of lists of tokens
    as input. This metric must accept two arguments when called, both
    of type `List[List[str]]`. The first is a predicted sequence for each item
    in the batch and the second is a gold sequence for each item in the batch.
initializer : ``InitializerApplicator``, optional
    An initialization strategy for the model weights.

### forward
```python
CopyNetSeq2Seq.forward(self, source_tokens:Dict[str, torch.LongTensor], source_token_ids:torch.Tensor, source_to_target:torch.Tensor, metadata:List[Dict[str, Any]], target_tokens:Dict[str, torch.LongTensor]=None, target_token_ids:torch.Tensor=None) -> Dict[str, torch.Tensor]
```

Make foward pass with decoder logic for producing the entire target sequence.

Parameters
----------
source_tokens : ``Dict[str, torch.LongTensor]``, required
    The output of `TextField.as_array()` applied on the source `TextField`. This will be
    passed through a `TextFieldEmbedder` and then through an encoder.
source_token_ids : ``torch.Tensor``, required
    Tensor containing IDs that indicate which source tokens match each other.
    Has shape: `(batch_size, trimmed_source_length)`.
source_to_target : ``torch.Tensor``, required
    Tensor containing vocab index of each source token with respect to the
    target vocab namespace. Shape: `(batch_size, trimmed_source_length)`.
metadata : ``List[Dict[str, Any]]``, required
    Metadata field that contains the original source tokens with key 'source_tokens'
    and any other meta fields. When 'target_tokens' is also passed, the metadata
    should also contain the original target tokens with key 'target_tokens'.
target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
    Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
    target tokens are also represented as a `TextField` which must contain a "tokens"
    key that uses single ids.
target_token_ids : ``torch.Tensor``, optional (default = None)
    A tensor of shape `(batch_size, target_sequence_length)` which indicates which
    tokens in the target sequence match tokens in the source sequence.

Returns
-------
Dict[str, torch.Tensor]

### take_search_step
```python
CopyNetSeq2Seq.take_search_step(self, last_predictions:torch.Tensor, state:Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

Take step during beam search.

This function is what gets passed to the `BeamSearch.search` method. It takes
predictions from the last timestep and the current state and outputs
the log probabilities assigned to tokens for the next timestep, as well as the updated
state.

Since we are predicting tokens out of the extended vocab (target vocab + all unique
tokens from the source sentence), this is a little more complicated that just
making a forward pass through the model. The output log probs will have
shape `(group_size, target_vocab_size + trimmed_source_length)` so that each
token in the target vocab and source sentence are assigned a probability.

Note that copy scores are assigned to each source token based on their position, not unique value.
So if a token appears more than once in the source sentence, it will have more than one score.
Further, if a source token is also part of the target vocab, its final score
will be the sum of the generation and copy scores. Therefore, in order to
get the score for all tokens in the extended vocab at this step,
we have to combine copy scores for re-occuring source tokens and potentially
add them to the generation scores for the matching token in the target vocab, if
there is one.

So we can break down the final log probs output as the concatenation of two
matrices, A: `(group_size, target_vocab_size)`, and B: `(group_size, trimmed_source_length)`.
Matrix A contains the sum of the generation score and copy scores (possibly 0)
for each target token. Matrix B contains left-over copy scores for source tokens
that do NOT appear in the target vocab, with zeros everywhere else. But since
a source token may appear more than once in the source sentence, we also have to
sum the scores for each appearance of each unique source token. So matrix B
actually only has non-zero values at the first occurence of each source token
that is not in the target vocab.

Parameters
----------
last_predictions : ``torch.Tensor``
    Shape: `(group_size,)`

state : ``Dict[str, torch.Tensor]``
    Contains all state tensors necessary to produce generation and copy scores
    for next step.

Notes
-----
`group_size` != `batch_size`. In fact, `group_size` = `batch_size * beam_size`.

### decode
```python
CopyNetSeq2Seq.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, Any]
```

Finalize predictions.

After a beam search, the predicted indices correspond to tokens in the target vocabulary
OR tokens in source sentence. Here we gather the actual tokens corresponding to
the indices.

### get_metrics
```python
CopyNetSeq2Seq.get_metrics(self, reset:bool=False) -> Dict[str, float]
```

Returns a dictionary of metrics. This method will be called by
:class:`allennlp.training.Trainer` in order to compute and use model metrics for early
stopping and model serialization.  We return an empty dictionary here rather than raising
as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
passed, as frequently a metric accumulator will have some state which should be reset
between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
should be populated during the call to ``forward``, with the
:class:`~allennlp.training.Metric` handling the accumulation of the metric until this
method is called.

