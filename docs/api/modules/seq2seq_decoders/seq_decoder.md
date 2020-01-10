# allennlp.modules.seq2seq_decoders.seq_decoder

## SeqDecoder
```python
SeqDecoder(self, target_embedder:allennlp.modules.token_embedders.embedding.Embedding) -> None
```

A ``SeqDecoder`` abstract class representing the entire decoder (embedding and neural network) of
a Seq2Seq architecture.
This is meant to be used with ``allennlp.models.encoder_decoder.composed_seq2seq.ComposedSeq2Seq``.

The implementation of this abstract class ideally uses a
decoder neural net ``allennlp.modules.seq2seq_decoders.decoder_net.DecoderNet`` for decoding.

The `default_implementation`
``allennlp.modules.seq2seq_decoders.seq_decoder.auto_regressive_seq_decoder.AutoRegressiveSeqDecoder``
covers most use cases. More likely that we will use the default implementation instead of creating a new
implementation.

Parameters
----------
target_embedder : ``Embedding``, required
    Embedder for target tokens. Needed in the base class to enable weight tying.

### default_implementation
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
### get_output_dim
```python
SeqDecoder.get_output_dim(self) -> int
```

The dimension of each timestep of the hidden state in the layer before final softmax.
Needed to check whether the model is compatible for embedding-final layer weight tying.

### get_metrics
```python
SeqDecoder.get_metrics(self, reset:bool=False) -> Dict[str, float]
```

The decoder is responsible for computing metrics using the target tokens.

### forward
```python
SeqDecoder.forward(self, encoder_out:Dict[str, torch.LongTensor], target_tokens:Union[Dict[str, torch.LongTensor], NoneType]=None) -> Dict[str, torch.Tensor]
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
SeqDecoder.post_process(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Post processing for converting raw outputs to prediction during inference.
The composing models such ``allennlp.models.encoder_decoders.composed_seq2seq.ComposedSeq2Seq``
can call this method when `decode` is called.

