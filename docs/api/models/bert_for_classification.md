# allennlp.models.bert_for_classification

## BertForClassification
```python
BertForClassification(self, vocab:allennlp.data.vocabulary.Vocabulary, bert_model:Union[str, pytorch_pretrained_bert.modeling.BertModel], dropout:float=0.0, num_labels:int=None, index:str='bert', label_namespace:str='labels', trainable:bool=True, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x1304d82e8>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

An AllenNLP Model that runs pretrained BERT,
takes the pooled output, and adds a Linear layer on top.
If you want an easy way to use BERT for classification, this is it.
Note that this is a somewhat non-AllenNLP-ish model architecture,
in that it essentially requires you to use the "bert-pretrained"
token indexer, rather than configuring whatever indexing scheme you like.

See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
for an example of what your config might look like.

Parameters
----------
vocab : ``Vocabulary``
bert_model : ``Union[str, BertModel]``
    The BERT model to be wrapped. If a string is provided, we will call
    ``BertModel.from_pretrained(bert_model)`` and use the result.
num_labels : ``int``, optional (default: None)
    How many output classes to predict. If not provided, we'll use the
    vocab_size for the ``label_namespace``.
index : ``str``, optional (default: "bert")
    The index of the token indexer that generates the BERT indices.
label_namespace : ``str``, optional (default : "labels")
    Used to determine the number of classes if ``num_labels`` is not supplied.
trainable : ``bool``, optional (default : True)
    If True, the weights of the pretrained BERT model will be updated during training.
    Otherwise, they will be frozen and only the final linear layer will be trained.
initializer : ``InitializerApplicator``, optional
    If provided, will be used to initialize the final linear layer *only*.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
BertForClassification.forward(self, tokens:Dict[str, torch.LongTensor], label:torch.IntTensor=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
tokens : Dict[str, torch.LongTensor]
    From a ``TextField`` (that has a bert-pretrained token indexer)
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
BertForClassification.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Does a simple argmax over the probabilities, converts index to string label, and
add ``"label"`` key to the dictionary with the result.

