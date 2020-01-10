# allennlp.models.event2mind

## Event2Mind
```python
Event2Mind(self, vocab:allennlp.data.vocabulary.Vocabulary, source_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, embedding_dropout:float, encoder:allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder, max_decoding_steps:int, beam_size:int=10, target_names:List[str]=None, target_namespace:str='tokens', target_embedding_dim:int=None, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This ``Event2Mind`` class is a :class:`Model` which takes an event
sequence, encodes it, and then uses the encoded representation to decode
several mental state sequences.

It is based on `the paper by Rashkin et al.
<https://www.semanticscholar.org/paper/Event2Mind/b89f8a9b2192a8f2018eead6b135ed30a1f2144d>`_

Parameters
----------
vocab : ``Vocabulary``, required
    Vocabulary containing source and target vocabularies. They may be under the same namespace
    (``tokens``) or the target tokens can have a different namespace, in which case it needs to
    be specified as ``target_namespace``.
source_embedder : ``TextFieldEmbedder``, required
    Embedder for source side sequences.
embedding_dropout: float, required
    The amount of dropout to apply after the source tokens have been embedded.
encoder : ``Seq2VecEncoder``, required
    The encoder of the "encoder/decoder" model.
max_decoding_steps : int, required
    Length of decoded sequences.
beam_size : int, optional (default = 10)
    The width of the beam search.
target_names : ``List[str]``, optional, (default = ['xintent', 'xreact', 'oreact'])
    Names of the target fields matching those in the ``Instance`` objects.
target_namespace : str, optional (default = 'tokens')
    If the target side vocabulary is different from the source side's, you need to specify the
    target's namespace here. If not, we'll assume it is "tokens", which is also the default
    choice for the source side, and this might cause them to share vocabularies.
target_embedding_dim : int, optional (default = source_embedding_dim)
    You can specify an embedding dimensionality for the target side. If not, we'll use the same
    value as the source embedder's.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
Event2Mind.forward(self, source:Dict[str, torch.LongTensor], **target_tokens:Dict[str, Dict[str, torch.LongTensor]]) -> Dict[str, torch.Tensor]
```

Decoder logic for producing the target sequences.

Parameters
----------
source : ``Dict[str, torch.LongTensor]``
    The output of ``TextField.as_array()`` applied on the source
    ``TextField``. This will be passed through a ``TextFieldEmbedder``
    and then through an encoder.
target_tokens : ``Dict[str, Dict[str, torch.LongTensor]]``:
    Dictionary from name to output of ``Textfield.as_array()`` applied
    on target ``TextField``. We assume that the target tokens are also
    represented as a ``TextField``.

### greedy_search
```python
Event2Mind.greedy_search(self, final_encoder_output:torch.LongTensor, target_tokens:Dict[str, torch.LongTensor], target_embedder:allennlp.modules.token_embedders.embedding.Embedding, decoder_cell:torch.nn.modules.rnn.GRUCell, output_projection_layer:torch.nn.modules.linear.Linear) -> torch.FloatTensor
```

Greedily produces a sequence using the provided ``decoder_cell``.
Returns the cross entropy between this sequence and ``target_tokens``.

Parameters
----------
final_encoder_output : ``torch.LongTensor``, required
    Vector produced by ``self._encoder``.
target_tokens : ``Dict[str, torch.LongTensor]``, required
    The output of ``TextField.as_array()`` applied on some target ``TextField``.
target_embedder : ``Embedding``, required
    Used to embed the target tokens.
decoder_cell : ``GRUCell``, required
    The recurrent cell used at each time step.
output_projection_layer : ``Linear``, required
    Linear layer mapping to the desired number of classes.

### greedy_predict
```python
Event2Mind.greedy_predict(self, final_encoder_output:torch.LongTensor, target_embedder:allennlp.modules.token_embedders.embedding.Embedding, decoder_cell:torch.nn.modules.rnn.GRUCell, output_projection_layer:torch.nn.modules.linear.Linear) -> torch.Tensor
```

Greedily produces a sequence using the provided ``decoder_cell``.
Returns the predicted sequence.

Parameters
----------
final_encoder_output : ``torch.LongTensor``, required
    Vector produced by ``self._encoder``.
target_embedder : ``Embedding``, required
    Used to embed the target tokens.
decoder_cell : ``GRUCell``, required
    The recurrent cell used at each time step.
output_projection_layer : ``Linear``, required
    Linear layer mapping to the desired number of classes.

### decode
```python
Event2Mind.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, List[List[str]]]
```

This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
within the ``forward`` method.

This method trims the output predictions to the first end symbol, replaces indices with
corresponding tokens, and adds fields for the tokens to the ``output_dict``.

### get_metrics
```python
Event2Mind.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

## StateDecoder
```python
StateDecoder(self, num_classes:int, input_dim:int, output_dim:int) -> None
```

Simple struct-like class for internal use.

