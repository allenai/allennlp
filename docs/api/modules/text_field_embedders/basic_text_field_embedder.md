# allennlp.modules.text_field_embedders.basic_text_field_embedder

## BasicTextFieldEmbedder
```python
BasicTextFieldEmbedder(self, token_embedders:Dict[str, allennlp.modules.token_embedders.token_embedder.TokenEmbedder], embedder_to_indexer_map:Dict[str, Union[List[str], Dict[str, str]]]=None, allow_unmatched_keys:bool=False) -> None
```

This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.  Each
``TokenEmbedder`` embeds or encodes the representation output from one
:class:`~allennlp.data.TokenIndexer`.  As the data produced by a
:class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
representations, we take ``TokenEmbedders`` with corresponding names.  Each ``TokenEmbedders``
embeds its input, and the result is concatenated in an arbitrary order.

Parameters
----------

token_embedders : ``Dict[str, TokenEmbedder]``, required.
    A dictionary mapping token embedder names to implementations.
    These names should match the corresponding indexer used to generate
    the tensor passed to the TokenEmbedder.
embedder_to_indexer_map : ``Dict[str, Union[List[str], Dict[str, str]]]``, optional, (default = None)
    Optionally, you can provide a mapping between the names of the TokenEmbedders that
    you are using to embed your TextField and an ordered list of indexer names which
    are needed for running it, or a mapping between the parameters which the
    ``TokenEmbedder.forward`` takes and the indexer names which are viewed as arguments.
    In most cases, your TokenEmbedder will only require a single tensor, because it is
    designed to run on the output of a single TokenIndexer. For example, the ELMo Token
    Embedder can be used in two modes, one of which requires both character ids and word
    ids for the same text. Note that the list of token indexer names is `ordered`,
    meaning that the tensors produced by the indexers will be passed to the embedders in
    the order you specify in this list. You can also use `null` in the configuration to
    set some specified parameters to None.
allow_unmatched_keys : ``bool``, optional (default = False)
    If True, then don't enforce the keys of the ``text_field_input`` to
    match those in ``token_embedders`` (useful if the mapping is specified
    via ``embedder_to_indexer_map``).

### get_output_dim
```python
BasicTextFieldEmbedder.get_output_dim(self) -> int
```

Returns the dimension of the vector representing each token in the output of this
``TextFieldEmbedder``.  This is `not` the shape of the returned tensor, but the last element of
that shape.

