# allennlp.models.bimpm

BiMPM (Bilateral Multi-Perspective Matching) model implementation.

## BiMpm
```python
BiMpm(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, matcher_word:allennlp.modules.bimpm_matching.BiMpmMatching, encoder1:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, matcher_forward1:allennlp.modules.bimpm_matching.BiMpmMatching, matcher_backward1:allennlp.modules.bimpm_matching.BiMpmMatching, encoder2:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, matcher_forward2:allennlp.modules.bimpm_matching.BiMpmMatching, matcher_backward2:allennlp.modules.bimpm_matching.BiMpmMatching, aggregator:allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder, classifier_feedforward:allennlp.modules.feedforward.FeedForward, dropout:float=0.1, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x138deeac8>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This ``Model`` implements BiMPM model described in `Bilateral Multi-Perspective Matching
for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017.
Also please refer to the `TensorFlow implementation <https://github.com/zhiguowang/BiMPM/>`_ and
`PyTorch implementation <https://github.com/galsang/BIMPM-pytorch>`_.

Parameters
----------
vocab : ``Vocabulary``
text_field_embedder : ``TextFieldEmbedder``
    Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
    model.
matcher_word : ``BiMpmMatching``
    BiMPM matching on the output of word embeddings of premise and hypothesis.
encoder1 : ``Seq2SeqEncoder``
    First encoder layer for the premise and hypothesis
matcher_forward1 : ``BiMPMMatching``
    BiMPM matching for the forward output of first encoder layer
matcher_backward1 : ``BiMPMMatching``
    BiMPM matching for the backward output of first encoder layer
encoder2 : ``Seq2SeqEncoder``
    Second encoder layer for the premise and hypothesis
matcher_forward2 : ``BiMPMMatching``
    BiMPM matching for the forward output of second encoder layer
matcher_backward2 : ``BiMPMMatching``
    BiMPM matching for the backward output of second encoder layer
aggregator : ``Seq2VecEncoder``
    Aggregator of all BiMPM matching vectors
classifier_feedforward : ``FeedForward``
    Fully connected layers for classification.
dropout : ``float``, optional (default=0.1)
    Dropout percentage to use.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    If provided, will be used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
BiMpm.forward(self, premise:Dict[str, torch.LongTensor], hypothesis:Dict[str, torch.LongTensor], label:torch.LongTensor=None, metadata:List[Dict[str, Any]]=None) -> Dict[str, torch.Tensor]
```


Parameters
----------
premise : Dict[str, torch.LongTensor]
    The premise from a ``TextField``
hypothesis : Dict[str, torch.LongTensor]
    The hypothesis from a ``TextField``
label : torch.LongTensor, optional (default = None)
    The label for the pair of the premise and the hypothesis
metadata : ``List[Dict[str, Any]]``, optional, (default = None)
    Additional information about the pair
Returns
-------
An output dictionary consisting of:

logits : torch.FloatTensor
    A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
    probabilities of the entailment label.
loss : torch.FloatTensor, optional
    A scalar loss to be optimised.

### decode
```python
BiMpm.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Converts indices to string labels, and adds a ``"label"`` key to the result.

### get_metrics
```python
BiMpm.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

