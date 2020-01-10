# allennlp.modules.seq2seq_decoders.auto_regressive_seq_decoder

## AutoRegressiveSeqDecoder
```python
AutoRegressiveSeqDecoder(self, vocab:allennlp.data.vocabulary.Vocabulary, decoder_net:allennlp.modules.seq2seq_decoders.decoder_net.DecoderNet, max_decoding_steps:int, target_embedder:allennlp.modules.token_embedders.embedding.Embedding, target_namespace:str='tokens', tie_output_embedding:bool=False, scheduled_sampling_ratio:float=0, label_smoothing_ratio:Union[float, NoneType]=None, beam_size:int=4, tensor_based_metric:allennlp.training.metrics.metric.Metric=None, token_based_metric:allennlp.training.metrics.metric.Metric=None) -> None
```

An autoregressive decoder that can be used for most seq2seq tasks.

Parameters
----------
vocab : ``Vocabulary``, required
    Vocabulary containing source and target vocabularies. They may be under the same namespace
    (`tokens`) or the target tokens can have a different namespace, in which case it needs to
    be specified as `target_namespace`.
decoder_net : ``DecoderNet``, required
    Module that contains implementation of neural network for decoding output elements
max_decoding_steps : ``int``, required
    Maximum length of decoded sequences.
target_embedder : ``Embedding``, required
    Embedder for target tokens.
target_namespace : ``str``, optional (default = 'tokens')
    If the target side vocabulary is different from the source side's, you need to specify the
    target's namespace here. If not, we'll assume it is "tokens", which is also the default
    choice for the source side, and this might cause them to share vocabularies.
beam_size : ``int``, optional (default = 4)
    Width of the beam for beam search.
tensor_based_metric : ``Metric``, optional (default = None)
    A metric to track on validation data that takes raw tensors when its called.
    This metric must accept two arguments when called: a batched tensor
    of predicted token indices, and a batched tensor of gold token indices.
token_based_metric : ``Metric``, optional (default = None)
    A metric to track on validation data that takes lists of lists of tokens
    as input. This metric must accept two arguments when called, both
    of type `List[List[str]]`. The first is a predicted sequence for each item
    in the batch and the second is a gold sequence for each item in the batch.
scheduled_sampling_ratio : ``float`` optional (default = 0)
    Defines ratio between teacher forced training and real output usage. If its zero
    (teacher forcing only) and `decoder_net`supports parallel decoding, we get the output
    predictions in a single forward pass of the `decoder_net`.

### take_step
```python
AutoRegressiveSeqDecoder.take_step(self, last_predictions:torch.Tensor, state:Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]
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

### get_metrics
```python
AutoRegressiveSeqDecoder.get_metrics(self, reset:bool=False) -> Dict[str, float]
```

The decoder is responsible for computing metrics using the target tokens.

### forward
```python
AutoRegressiveSeqDecoder.forward(self, encoder_out:Dict[str, torch.LongTensor], target_tokens:Dict[str, torch.LongTensor]=None) -> Dict[str, torch.Tensor]
```

Decoding from encoded states to sequence of outputs
also computes loss if ``target_tokens`` are given.

Parameters
----------
encoder_out : ``Dict[str, torch.LongTensor]``, required
    Dictionary with encoded state, ideally containing the encoded vectors and the
    source mask.
target_tokens : ``Dict[str, torch.LongTensor]``, optional
    The output of `TextField.as_array()` applied on the target `TextField`.


### post_process
```python
AutoRegressiveSeqDecoder.post_process(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

This method trims the output predictions to the first end symbol, replaces indices with
corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.

