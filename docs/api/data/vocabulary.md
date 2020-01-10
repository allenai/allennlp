# allennlp.data.vocabulary

A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.

## pop_max_vocab_size
```python
pop_max_vocab_size(params:allennlp.common.params.Params) -> Union[int, Dict[str, int]]
```

max_vocab_size limits the size of the vocabulary, not including the @@UNKNOWN@@ token.

max_vocab_size is allowed to be either an int or a Dict[str, int] (or nothing).
But it could also be a string representing an int (in the case of environment variable
substitution). So we need some complex logic to handle it.

## Vocabulary
```python
Vocabulary(self, counter:Dict[str, Dict[str, int]]=None, min_count:Dict[str, int]=None, max_vocab_size:Union[int, Dict[str, int]]=None, non_padded_namespaces:Iterable[str]=('*tags', '*labels'), pretrained_files:Union[Dict[str, str], NoneType]=None, only_include_pretrained_words:bool=False, tokens_to_add:Dict[str, List[str]]=None, min_pretrained_embeddings:Dict[str, int]=None, padding_token:Union[str, NoneType]='@@PADDING@@', oov_token:Union[str, NoneType]='@@UNKNOWN@@') -> None
```

A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.

Vocabularies are fit to a particular dataset, which we use to decide which tokens are
in-vocabulary.

Vocabularies also allow for several different namespaces, so you can have separate indices for
'a' as a word, and 'a' as a character, for instance, and so we can use this object to also map
tag and label strings to indices, for a unified :class:`~.fields.field.Field` API.  Most of the
methods on this class allow you to pass in a namespace; by default we use the 'tokens'
namespace, and you can omit the namespace argument everywhere and just use the default.

Parameters
----------
counter : ``Dict[str, Dict[str, int]]``, optional (default=``None``)
    A collection of counts from which to initialize this vocabulary.  We will examine the
    counts and, together with the other parameters to this class, use them to decide which
    words are in-vocabulary.  If this is ``None``, we just won't initialize the vocabulary with
    anything.
min_count : ``Dict[str, int]``, optional (default=None)
    When initializing the vocab from a counter, you can specify a minimum count, and every
    token with a count less than this will not be added to the dictionary.  These minimum
    counts are `namespace-specific`, so you can specify different minimums for labels versus
    words tokens, for example.  If a namespace does not have a key in the given dictionary, we
    will add all seen tokens to that namespace.
max_vocab_size : ``Union[int, Dict[str, int]]``, optional (default=``None``)
    If you want to cap the number of tokens in your vocabulary, you can do so with this
    parameter.  If you specify a single integer, every namespace will have its vocabulary fixed
    to be no larger than this.  If you specify a dictionary, then each namespace in the
    ``counter`` can have a separate maximum vocabulary size.  Any missing key will have a value
    of ``None``, which means no cap on the vocabulary size.
non_padded_namespaces : ``Iterable[str]``, optional
    By default, we assume you are mapping word / character tokens to integers, and so you want
    to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
    mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
    indices for padding and out-of-vocabulary tokens.  Use this field to specify which
    namespaces should `not` have padding and OOV tokens added.

    The format of each element of this is either a string, which must match field names
    exactly,  or ``*`` followed by a string, which we match as a suffix against field names.

    We try to make the default here reasonable, so that you don't have to think about this.
    The default is ``("*tags", "*labels")``, so as long as your namespace ends in "tags" or
    "labels" (which is true by default for all tag and label fields in this code), you don't
    have to specify anything here.
pretrained_files : ``Dict[str, str]``, optional
    If provided, this map specifies the path to optional pretrained embedding files for each
    namespace. This can be used to either restrict the vocabulary to only words which appear
    in this file, or to ensure that any words in this file are included in the vocabulary
    regardless of their count, depending on the value of ``only_include_pretrained_words``.
    Words which appear in the pretrained embedding file but not in the data are NOT included
    in the Vocabulary.
min_pretrained_embeddings : ``Dict[str, int]``, optional
    If provided, specifies for each namespace a minimum number of lines (typically the
    most common words) to keep from pretrained embedding files, even for words not
    appearing in the data.
only_include_pretrained_words : ``bool``, optional (default=False)
    This defines the strategy for using any pretrained embedding files which may have been
    specified in ``pretrained_files``. If False, an inclusive strategy is used: and words
    which are in the ``counter`` and in the pretrained file are added to the ``Vocabulary``,
    regardless of whether their count exceeds ``min_count`` or not. If True, we use an
    exclusive strategy: words are only included in the Vocabulary if they are in the pretrained
    embedding file (their count must still be at least ``min_count``).
tokens_to_add : ``Dict[str, List[str]]``, optional (default=None)
    If given, this is a list of tokens to add to the vocabulary, keyed by the namespace to add
    the tokens to.  This is a way to be sure that certain items appear in your vocabulary,
    regardless of any other vocabulary computation.
padding_token : ``str``,  optional (default=DEFAULT_PADDING_TOKEN)
    If given, this the string used for padding.
oov_token : ``str``,  optional (default=DEFAULT_OOV_TOKEN)
    If given, this the string used for the out of vocabulary (OOVs) tokens.

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
### save_to_files
```python
Vocabulary.save_to_files(self, directory:str) -> None
```

Persist this Vocabulary to files so it can be reloaded later.
Each namespace corresponds to one file.

Parameters
----------
directory : ``str``
    The directory where we save the serialized vocabulary.

### from_files
```python
Vocabulary.from_files(directory:str, padding_token:Union[str, NoneType]='@@PADDING@@', oov_token:Union[str, NoneType]='@@UNKNOWN@@') -> 'Vocabulary'
```

Loads a ``Vocabulary`` that was serialized using ``save_to_files``.

Parameters
----------
directory : ``str``
    The directory containing the serialized vocabulary.

### set_from_file
```python
Vocabulary.set_from_file(self, filename:str, is_padded:bool=True, oov_token:str='@@UNKNOWN@@', namespace:str='tokens')
```

If you already have a vocabulary file for a trained model somewhere, and you really want to
use that vocabulary file instead of just setting the vocabulary from a dataset, for
whatever reason, you can do that with this method.  You must specify the namespace to use,
and we assume that you want to use padding and OOV tokens for this.

Parameters
----------
filename : ``str``
    The file containing the vocabulary to load.  It should be formatted as one token per
    line, with nothing else in the line.  The index we assign to the token is the line
    number in the file (1-indexed if ``is_padded``, 0-indexed otherwise).  Note that this
    file should contain the OOV token string!
is_padded : ``bool``, optional (default=True)
    Is this vocabulary padded?  For token / word / character vocabularies, this should be
    ``True``; while for tag or label vocabularies, this should typically be ``False``.  If
    ``True``, we add a padding token with index 0, and we enforce that the ``oov_token`` is
    present in the file.
oov_token : ``str``, optional (default=DEFAULT_OOV_TOKEN)
    What token does this vocabulary use to represent out-of-vocabulary characters?  This
    must show up as a line in the vocabulary file.  When we find it, we replace
    ``oov_token`` with ``self._oov_token``, because we only use one OOV token across
    namespaces.
namespace : ``str``, optional (default="tokens")
    What namespace should we overwrite with this vocab file?

### from_instances
```python
Vocabulary.from_instances(instances:Iterable[_ForwardRef('adi.Instance')], min_count:Dict[str, int]=None, max_vocab_size:Union[int, Dict[str, int]]=None, non_padded_namespaces:Iterable[str]=('*tags', '*labels'), pretrained_files:Union[Dict[str, str], NoneType]=None, only_include_pretrained_words:bool=False, tokens_to_add:Dict[str, List[str]]=None, min_pretrained_embeddings:Dict[str, int]=None, padding_token:Union[str, NoneType]='@@PADDING@@', oov_token:Union[str, NoneType]='@@UNKNOWN@@') -> 'Vocabulary'
```

Constructs a vocabulary given a collection of `Instances` and some parameters.
We count all of the vocabulary items in the instances, then pass those counts
and the other parameters, to :func:`__init__`.  See that method for a description
of what the other parameters do.

### from_params
```python
Vocabulary.from_params(params:allennlp.common.params.Params, instances:Iterable[_ForwardRef('adi.Instance')]=None)
```

There are two possible ways to build a vocabulary; from a
collection of instances, using :func:`Vocabulary.from_instances`, or
from a pre-saved vocabulary, using :func:`Vocabulary.from_files`.
You can also extend pre-saved vocabulary with collection of instances
using this method. This method wraps these options, allowing their
specification from a ``Params`` object, generated from a JSON
configuration file.

Parameters
----------
params: Params, required.
instances: Iterable['adi.Instance'], optional
    If ``params`` doesn't contain a ``directory_path`` key,
    the ``Vocabulary`` can be built directly from a collection of
    instances (i.e. a dataset). If ``extend`` key is set False,
    dataset instances will be ignored and final vocabulary will be
    one loaded from ``directory_path``. If ``extend`` key is set True,
    dataset instances will be used to extend the vocabulary loaded
    from ``directory_path`` and that will be final vocabulary used.

Returns
-------
A ``Vocabulary``.

### extend_from_instances
```python
Vocabulary.extend_from_instances(self, params:allennlp.common.params.Params, instances:Iterable[_ForwardRef('adi.Instance')]=()) -> None
```

Extends an already generated vocabulary using a collection of instances.

### is_padded
```python
Vocabulary.is_padded(self, namespace:str) -> bool
```

Returns whether or not there are padding and OOV tokens added to the given namespace.

### add_token_to_namespace
```python
Vocabulary.add_token_to_namespace(self, token:str, namespace:str='tokens') -> int
```

Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
the token.

### add_tokens_to_namespace
```python
Vocabulary.add_tokens_to_namespace(self, tokens:List[str], namespace:str='tokens') -> List[int]
```

Adds ``tokens`` to the index, if they are not already present.  Either way, we return the
indices of the tokens in the order that they were given.

