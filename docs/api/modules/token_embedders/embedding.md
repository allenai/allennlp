# allennlp.modules.token_embedders.embedding

## Embedding
```python
Embedding(self, num_embeddings:int, embedding_dim:int, projection_dim:int=None, weight:torch.FloatTensor=None, padding_index:int=None, trainable:bool=True, max_norm:float=None, norm_type:float=2.0, scale_grad_by_freq:bool=False, sparse:bool=False, vocab_namespace:str=None, pretrained_file:str=None) -> None
```

A more featureful embedding module than the default in Pytorch.  Adds the ability to:

    1. embed higher-order inputs
    2. pre-specify the weight matrix
    3. use a non-trainable embedding
    4. project the resultant embeddings to some other dimension (which only makes sense with
       non-trainable embeddings).
    5. build all of this easily ``from_params``

Note that if you are using our data API and are trying to embed a
:class:`~allennlp.data.fields.TextField`, you should use a
:class:`~allennlp.modules.TextFieldEmbedder` instead of using this directly.

Parameters
----------
num_embeddings : ``int``
    Size of the dictionary of embeddings (vocabulary size).
embedding_dim : ``int``
    The size of each embedding vector.
projection_dim : ``int``, (optional, default=None)
    If given, we add a projection layer after the embedding layer.  This really only makes
    sense if ``trainable`` is ``False``.
weight : ``torch.FloatTensor``, (optional, default=None)
    A pre-initialised weight matrix for the embedding lookup, allowing the use of
    pretrained vectors.
padding_index : ``int``, (optional, default=None)
    If given, pads the output with zeros whenever it encounters the index.
trainable : ``bool``, (optional, default=True)
    Whether or not to optimize the embedding parameters.
max_norm : ``float``, (optional, default=None)
    If given, will renormalize the embeddings to always have a norm lesser than this
norm_type : ``float``, (optional, default=2)
    The p of the p-norm to compute for the max_norm option
scale_grad_by_freq : ``bool``, (optional, default=False)
    If given, this will scale gradients by the frequency of the words in the mini-batch.
sparse : ``bool``, (optional, default=False)
    Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
vocab_namespace : ``str``, (optional, default=None)
    In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
    extended according to the size of extended-vocabulary. To be able to know how much to
    extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
    construct it in the original training. We store vocab_namespace used during the original
    training as an attribute, so that it can be retrieved during fine-tuning.
pretrained_file : ``str``, (optional, default=None)
    Used to keep track of what is the source of the weights and loading more embeddings at test time.
    **It does not load the weights from this pretrained_file.** For that purpose, use
    ``Embedding.from_params``.

Returns
-------
An Embedding module.

### get_output_dim
```python
Embedding.get_output_dim(self) -> int
```

Returns the final output dimension that this ``TokenEmbedder`` uses to represent each
token.  This is `not` the shape of the returned tensor, but the last element of that shape.

### forward
```python
Embedding.forward(self, inputs)
```
Defines the computation performed at every call.

Should be overridden by all subclasses.

.. note::
    Although the recipe for forward pass needs to be defined within
    this function, one should call the :class:`Module` instance afterwards
    instead of this since the former takes care of running the
    registered hooks while the latter silently ignores them.

### extend_vocab
```python
Embedding.extend_vocab(self, extended_vocab:allennlp.data.vocabulary.Vocabulary, vocab_namespace:str=None, extension_pretrained_file:str=None, model_path:str=None)
```

Extends the embedding matrix according to the extended vocabulary.
If extension_pretrained_file is available, it will be used for initializing the new words
embeddings in the extended vocabulary; otherwise we will check if _pretrained_file attribute
is already available. If none is available, they will be initialized with xavier uniform.

Parameters
----------
extended_vocab : ``Vocabulary``
    Vocabulary extended from original vocabulary used to construct
    this ``Embedding``.
vocab_namespace : ``str``, (optional, default=None)
    In case you know what vocab_namespace should be used for extension, you
    can pass it. If not passed, it will check if vocab_namespace used at the
    time of ``Embedding`` construction is available. If so, this namespace
    will be used or else extend_vocab will be a no-op.
extension_pretrained_file : ``str``, (optional, default=None)
    A file containing pretrained embeddings can be specified here. It can be
    the path to a local file or an URL of a (cached) remote file. Check format
    details in ``from_params`` of ``Embedding`` class.
model_path : ``str``, (optional, default=None)
    Path traversing the model attributes upto this embedding module.
    Eg. "_text_field_embedder.token_embedder_tokens". This is only useful
    to give helpful error message when extend_vocab is implicitly called
    by fine-tune or any other command.

### from_params
```python
Embedding.from_params(vocab:allennlp.data.vocabulary.Vocabulary, params:allennlp.common.params.Params) -> 'Embedding'
```

We need the vocabulary here to know how many items we need to embed, and we look for a
``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.  If
you know beforehand exactly how many embeddings you need, or aren't using a vocabulary
mapping for the things getting embedded here, then you can pass in the ``num_embeddings``
key directly, and the vocabulary will be ignored.

In the configuration file, a file containing pretrained embeddings can be specified
using the parameter ``"pretrained_file"``.
It can be the path to a local file or an URL of a (cached) remote file.
Two formats are supported:

    * hdf5 file - containing an embedding matrix in the form of a torch.Tensor;

    * text file - an utf-8 encoded text file with space separated fields::

            [word] [dim 1] [dim 2] ...

      The text file can eventually be compressed with gzip, bz2, lzma or zip.
      You can even select a single file inside an archive containing multiple files
      using the URI::

            "(archive_uri)#file_path_inside_the_archive"

      where ``archive_uri`` can be a file system path or a URL. For example::

            "(https://nlp.stanford.edu/data/glove.twitter.27B.zip)#glove.twitter.27B.200d.txt"

## EmbeddingsFileURI
```python
EmbeddingsFileURI(self, /, *args, **kwargs)
```
EmbeddingsFileURI(main_file_uri, path_inside_archive)
### main_file_uri
Alias for field number 0
### path_inside_archive
Alias for field number 1
## EmbeddingsTextFile
```python
EmbeddingsTextFile(self, file_uri:str, encoding:str='utf-8', cache_dir:str=None) -> None
```

Utility class for opening embeddings text files. Handles various compression formats,
as well as context management.

Parameters
----------
file_uri : ``str``
    It can be:

    * a file system path or a URL of an eventually compressed text file or a zip/tar archive
      containing a single file.
    * URI of the type ``(archive_path_or_url)#file_path_inside_archive`` if the text file
      is contained in a multi-file archive.

encoding : ``str``
cache_dir : ``str``

### DEFAULT_ENCODING
str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.
