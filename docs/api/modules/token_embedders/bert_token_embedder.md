# allennlp.modules.token_embedders.bert_token_embedder

A ``TokenEmbedder`` which uses one of the BERT models
(https://github.com/google-research/bert)
to produce embeddings.

At its core it uses Hugging Face's PyTorch implementation
(https://github.com/huggingface/pytorch-pretrained-BERT),
so thanks to them!

## PretrainedBertModel
```python
PretrainedBertModel(self, /, *args, **kwargs)
```

In some instances you may want to load the same BERT model twice
(e.g. to use as a token embedder and also as a pooling layer).
This factory provides a cache so that you don't actually have to load the model twice.

## BertEmbedder
```python
BertEmbedder(self, bert_model:pytorch_pretrained_bert.modeling.BertModel, top_layer_only:bool=False, max_pieces:int=512, num_start_tokens:int=1, num_end_tokens:int=1, scalar_mix_parameters:List[float]=None) -> None
```

A ``TokenEmbedder`` that produces BERT embeddings for your tokens.
Should be paired with a ``BertIndexer``, which produces wordpiece ids.

Most likely you probably want to use ``PretrainedBertEmbedder``
for one of the named pretrained models, not this base class.

Parameters
----------
bert_model : ``BertModel``
    The BERT model being wrapped.
top_layer_only : ``bool``, optional (default = ``False``)
    If ``True``, then only return the top layer instead of apply the scalar mix.
max_pieces : ``int``, optional (default: 512)
    The BERT embedder uses positional embeddings and so has a corresponding
    maximum length for its input ids. Assuming the inputs are windowed
    and padded appropriately by this length, the embedder will split them into a
    large batch, feed them into BERT, and recombine the output as if it was a
    longer sequence.
num_start_tokens : ``int``, optional (default: 1)
    The number of starting special tokens input to BERT (usually 1, i.e., [CLS])
num_end_tokens : ``int``, optional (default: 1)
    The number of ending tokens input to BERT (usually 1, i.e., [SEP])
scalar_mix_parameters : ``List[float]``, optional, (default = None)
    If not ``None``, use these scalar mix parameters to weight the representations
    produced by different layers. These mixing weights are not updated during
    training.

### forward
```python
BertEmbedder.forward(self, input_ids:torch.LongTensor, offsets:torch.LongTensor=None, token_type_ids:torch.LongTensor=None) -> torch.Tensor
```

Parameters
----------
input_ids : ``torch.LongTensor``
    The (batch_size, ..., max_sequence_length) tensor of wordpiece ids.
offsets : ``torch.LongTensor``, optional
    The BERT embeddings are one per wordpiece. However it's possible/likely
    you might want one per original token. In that case, ``offsets``
    represents the indices of the desired wordpiece for each original token.
    Depending on how your token indexer is configured, this could be the
    position of the last wordpiece for each token, or it could be the position
    of the first wordpiece for each token.

    For example, if you had the sentence "Definitely not", and if the corresponding
    wordpieces were ["Def", "##in", "##ite", "##ly", "not"], then the input_ids
    would be 5 wordpiece ids, and the "last wordpiece" offsets would be [3, 4].
    If offsets are provided, the returned tensor will contain only the wordpiece
    embeddings at those positions, and (in particular) will contain one embedding
    per token. If offsets are not provided, the entire tensor of wordpiece embeddings
    will be returned.
token_type_ids : ``torch.LongTensor``, optional
    If an input consists of two sentences (as in the BERT paper),
    tokens from the first sentence should have type 0 and tokens from
    the second sentence should have type 1.  If you don't provide this
    (the default BertIndexer doesn't) then it's assumed to be all 0s.

## PretrainedBertEmbedder
```python
PretrainedBertEmbedder(self, pretrained_model:str, requires_grad:bool=False, top_layer_only:bool=False, scalar_mix_parameters:List[float]=None) -> None
```

Parameters
----------
pretrained_model : ``str``
    Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
    or the path to the .tar.gz file with the model weights.

    If the name is a key in the list of pretrained models at
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py#L41
    the corresponding path will be used; otherwise it will be interpreted as a path or URL.
requires_grad : ``bool``, optional (default = False)
    If True, compute gradient of BERT parameters for fine tuning.
top_layer_only : ``bool``, optional (default = ``False``)
    If ``True``, then only return the top layer instead of apply the scalar mix.
scalar_mix_parameters : ``List[float]``, optional, (default = None)
    If not ``None``, use these scalar mix parameters to weight the representations
    produced by different layers. These mixing weights are not updated during
    training.

