# allennlp.models.basic_classifier

## BasicClassifier
```python
BasicClassifier(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, seq2vec_encoder:allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder, seq2seq_encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder=None, feedforward:Union[allennlp.modules.feedforward.FeedForward, NoneType]=None, dropout:float=None, num_labels:int=None, label_namespace:str='labels', initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x13b94acc0>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This ``Model`` implements a basic text classifier. After embedding the text into
a text field, we will optionally encode the embeddings with a ``Seq2SeqEncoder``. The
resulting sequence is pooled using a ``Seq2VecEncoder`` and then passed to
a linear classification layer, which projects into the label space. If a
``Seq2SeqEncoder`` is not provided, we will pass the embedded text directly to the
``Seq2VecEncoder``.

Parameters
----------
vocab : ``Vocabulary``
text_field_embedder : ``TextFieldEmbedder``
    Used to embed the input text into a ``TextField``
seq2seq_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
    Optional Seq2Seq encoder layer for the input text.
seq2vec_encoder : ``Seq2VecEncoder``
    Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
    will pool its output. Otherwise, this encoder will operate directly on the output
    of the `text_field_embedder`.
feedforward : ``FeedForward``, optional, (default = None).
    An optional feedforward layer to apply after the seq2vec_encoder.
dropout : ``float``, optional (default = ``None``)
    Dropout percentage to use.
num_labels : ``int``, optional (default = ``None``)
    Number of labels to project to in classification layer. By default, the classification layer will
    project to the size of the vocabulary namespace corresponding to labels.
label_namespace : ``str``, optional (default = "labels")
    Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    If provided, will be used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
BasicClassifier.forward(self, tokens:Dict[str, torch.LongTensor], label:torch.IntTensor=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
tokens : Dict[str, torch.LongTensor]
    From a ``TextField``
label : torch.IntTensor, optional (default = None)
    From a ``LabelField``

Returns
-------
An output dictionary consisting of:

logits : torch.FloatTensor
    A tensor of shape ``(batch_size, num_labels)`` representing
    unnormalized log probabilities of the label.
probs : torch.FloatTensor
    A tensor of shape ``(batch_size, num_labels)`` representing
    probabilities of the label.
loss : torch.FloatTensor, optional
    A scalar loss to be optimised.

### decode
```python
BasicClassifier.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Does a simple argmax over the probabilities, converts index to string label, and
add ``"label"`` key to the dictionary with the result.

