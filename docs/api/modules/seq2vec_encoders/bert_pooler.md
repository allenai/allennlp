# allennlp.modules.seq2vec_encoders.bert_pooler

## BertPooler
```python
BertPooler(self, pretrained_model:Union[str, pytorch_pretrained_bert.modeling.BertModel], requires_grad:bool=True, dropout:float=0.0) -> None
```

The pooling layer at the end of the BERT model. This returns an embedding for the
[CLS] token, after passing it through a non-linear tanh activation; the non-linear layer
is also part of the BERT model. If you want to use the pretrained BERT model
to build a classifier and you want to use the AllenNLP token-indexer ->
token-embedder -> seq2vec encoder setup, this is the Seq2VecEncoder to use.
(For example, if you want to experiment with other embedding / encoding combinations.)

If you just want to train a BERT classifier, it's simpler to just use the
``BertForClassification`` model.

Parameters
----------
pretrained_model : ``Union[str, BertModel]``, required
    The pretrained BERT model to use. If this is a string,
    we will call ``BertModel.from_pretrained(pretrained_model)``
    and use that.
requires_grad : ``bool``, optional, (default = True)
    If True, the weights of the pooler will be updated during training.
    Otherwise they will not.
dropout : ``float``, optional, (default = 0.0)
    Amount of dropout to apply after pooling

### get_input_dim
```python
BertPooler.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2VecEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
BertPooler.get_output_dim(self) -> int
```

Returns the dimension of the final vector output by this ``Seq2VecEncoder``.  This is `not`
the shape of the returned tensor, but the last element of that shape.

