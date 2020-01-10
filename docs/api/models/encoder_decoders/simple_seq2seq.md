# allennlp.models.encoder_decoders.simple_seq2seq

## SimpleSeq2Seq
```python
SimpleSeq2Seq(self, vocab:allennlp.data.vocabulary.Vocabulary, source_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, max_decoding_steps:int, attention:allennlp.modules.attention.attention.Attention=None, attention_function:allennlp.modules.similarity_functions.similarity_function.SimilarityFunction=None, beam_size:int=None, target_namespace:str='tokens', target_embedding_dim:int=None, scheduled_sampling_ratio:float=0.0, use_bleu:bool=True) -> None
```

This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
uses the encoded representations to decode another sequence.  You can use this as the basis for
a neural machine translation system, an abstractive summarization system, or any other common
seq2seq problem.  The model here is simple, but should be a decent starting place for
implementing recent models for these tasks.

Parameters
----------
vocab : ``Vocabulary``, required
    Vocabulary containing source and target vocabularies. They may be under the same namespace
    (`tokens`) or the target tokens can have a different namespace, in which case it needs to
    be specified as `target_namespace`.
source_embedder : ``TextFieldEmbedder``, required
    Embedder for source side sequences
encoder : ``Seq2SeqEncoder``, required
    The encoder of the "encoder/decoder" model
max_decoding_steps : ``int``
    Maximum length of decoded sequences.
target_namespace : ``str``, optional (default = 'tokens')
    If the target side vocabulary is different from the source side's, you need to specify the
    target's namespace here. If not, we'll assume it is "tokens", which is also the default
    choice for the source side, and this might cause them to share vocabularies.
target_embedding_dim : ``int``, optional (default = source_embedding_dim)
    You can specify an embedding dimensionality for the target side. If not, we'll use the same
    value as the source embedder's.
attention : ``Attention``, optional (default = None)
    If you want to use attention to get a dynamic summary of the encoder outputs at each step
    of decoding, this is the function used to compute similarity between the decoder hidden
    state and encoder outputs.
attention_function : ``SimilarityFunction``, optional (default = None)
    This is if you want to use the legacy implementation of attention. This will be deprecated
    since it consumes more memory than the specialized attention modules.
beam_size : ``int``, optional (default = None)
    Width of the beam for beam search. If not specified, greedy decoding is used.
scheduled_sampling_ratio : ``float``, optional (default = 0.)
    At each timestep during training, we sample a random number between 0 and 1, and if it is
    not less than this value, we use the ground truth labels for the whole batch. Else, we use
    the predictions from the previous time step for the whole batch. If this value is 0.0
    (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
    using target side ground truth labels.  See the following paper for more information:
    `Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
    2015 <https://arxiv.org/abs/1506.03099>`_.
use_bleu : ``bool``, optional (default = True)
    If True, the BLEU metric will be calculated during validation.

### take_step
```python
SimpleSeq2Seq.take_step(self, last_predictions:torch.Tensor, state:Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
```

Take a decoding step. This is called by the beam search class.

Parameters
----------
last_predictions : ``torch.Tensor``
    A tensor of shape ``(group_size,)``, which gives the indices of the predictions
    during the last time step.
state : ``Dict[str, torch.Tensor]``
    A dictionary of tensors that contain the current state information
    needed to predict the next step, which includes the encoder outputs,
    the source mask, and the decoder hidden state and context. Each of these
    tensors has shape ``(group_size, *)``, where ``*`` can be any other number
    of dimensions.

Returns
-------
Tuple[torch.Tensor, Dict[str, torch.Tensor]]
    A tuple of ``(log_probabilities, updated_state)``, where ``log_probabilities``
    is a tensor of shape ``(group_size, num_classes)`` containing the predicted
    log probability of each class for the next step, for each item in the group,
    while ``updated_state`` is a dictionary of tensors containing the encoder outputs,
    source mask, and updated decoder hidden state and context.

Notes
-----
    We treat the inputs as a batch, even though ``group_size`` is not necessarily
    equal to ``batch_size``, since the group may contain multiple states
    for each source sentence in the batch.

### forward
```python
SimpleSeq2Seq.forward(self, source_tokens:Dict[str, torch.LongTensor], target_tokens:Dict[str, torch.LongTensor]=None) -> Dict[str, torch.Tensor]
```

Make foward pass with decoder logic for producing the entire target sequence.

Parameters
----------
source_tokens : ``Dict[str, torch.LongTensor]``
   The output of `TextField.as_array()` applied on the source `TextField`. This will be
   passed through a `TextFieldEmbedder` and then through an encoder.
target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
   Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
   target tokens are also represented as a `TextField`.

Returns
-------
Dict[str, torch.Tensor]

### decode
```python
SimpleSeq2Seq.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Finalize predictions.

This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
within the ``forward`` method.

This method trims the output predictions to the first end symbol, replaces indices with
corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.

### get_metrics
```python
SimpleSeq2Seq.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

