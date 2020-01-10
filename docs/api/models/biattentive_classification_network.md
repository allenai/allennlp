# allennlp.models.biattentive_classification_network

## BiattentiveClassificationNetwork
```python
BiattentiveClassificationNetwork(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, embedding_dropout:float, pre_encode_feedforward:allennlp.modules.feedforward.FeedForward, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, integrator:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, integrator_dropout:float, output_layer:Union[allennlp.modules.feedforward.FeedForward, allennlp.modules.maxout.Maxout], elmo:allennlp.modules.elmo.Elmo, use_input_elmo:bool=False, use_integrator_output_elmo:bool=False, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x12fdb7630>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This class implements the Biattentive Classification Network model described
in section 5 of `Learned in Translation: Contextualized Word Vectors (NIPS 2017)
<https://arxiv.org/abs/1708.00107>`_ for text classification. We assume we're
given a piece of text, and we predict some output label.

At a high level, the model starts by embedding the tokens and running them through
a feed-forward neural net (``pre_encode_feedforward``). Then, we encode these
representations with a ``Seq2SeqEncoder`` (``encoder``). We run biattention
on the encoder output representations (self-attention in this case, since
the two representations that typically go into biattention are identical) and
get out an attentive vector representation of the text. We combine this text
representation with the encoder outputs computed earlier, and then run this through
yet another ``Seq2SeqEncoder`` (the ``integrator``). Lastly, we take the output of the
integrator and max, min, mean, and self-attention pool to create a final representation,
which is passed through a maxout network or some feed-forward layers
to output a classification (``output_layer``).

Parameters
----------
vocab : ``Vocabulary``, required
    A Vocabulary, required in order to compute sizes for input/output projections.
text_field_embedder : ``TextFieldEmbedder``, required
    Used to embed the ``tokens`` ``TextField`` we get as input to the model.
embedding_dropout : ``float``
    The amount of dropout to apply on the embeddings.
pre_encode_feedforward : ``FeedForward``
    A feedforward network that is run on the embedded tokens before they
    are passed to the encoder.
encoder : ``Seq2SeqEncoder``
    The encoder to use on the tokens.
integrator : ``Seq2SeqEncoder``
    The encoder to use when integrating the attentive text encoding
    with the token encodings.
integrator_dropout : ``float``
    The amount of dropout to apply on integrator output.
output_layer : ``Union[Maxout, FeedForward]``
    The maxout or feed forward network that takes the final representations and produces
    a classification prediction.
elmo : ``Elmo``, optional (default=``None``)
    If provided, will be used to concatenate pretrained ELMo representations to
    either the integrator output (``use_integrator_output_elmo``) or the
    input (``use_input_elmo``).
use_input_elmo : ``bool`` (default=``False``)
    If true, concatenate pretrained ELMo representations to the input vectors.
use_integrator_output_elmo : ``bool`` (default=``False``)
    If true, concatenate pretrained ELMo representations to the integrator output.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
BiattentiveClassificationNetwork.forward(self, tokens:Dict[str, torch.LongTensor], label:torch.LongTensor=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
tokens : Dict[str, torch.LongTensor], required
    The output of ``TextField.as_array()``.
label : torch.LongTensor, optional (default = None)
    A variable representing the label for each instance in the batch.
Returns
-------
An output dictionary consisting of:
class_probabilities : torch.FloatTensor
    A tensor of shape ``(batch_size, num_classes)`` representing a
    distribution over the label classes for each instance.
loss : torch.FloatTensor, optional
    A scalar loss to be optimised.

### decode
```python
BiattentiveClassificationNetwork.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Does a simple argmax over the class probabilities, converts indices to string labels, and
adds a ``"label"`` key to the dictionary with the result.

### get_metrics
```python
BiattentiveClassificationNetwork.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

