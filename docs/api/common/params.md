# allennlp.common.params

The :class:`~allennlp.common.params.Params` class represents a dictionary of
parameters (e.g. for configuring a model), with added functionality around
logging and validation.

## infer_and_cast
```python
infer_and_cast(value:Any)
```

In some cases we'll be feeding params dicts to functions we don't own;
for example, PyTorch optimizers. In that case we can't use ``pop_int``
or similar to force casts (which means you can't specify ``int`` parameters
using environment variables). This function takes something that looks JSON-like
and recursively casts things that look like (bool, int, float) to (bool, int, float).

## unflatten
```python
unflatten(flat_dict:Dict[str, Any]) -> Dict[str, Any]
```

Given a "flattened" dict with compound keys, e.g.
    {"a.b": 0}
unflatten it:
    {"a": {"b": 0}}

## with_fallback
```python
with_fallback(preferred:Dict[str, Any], fallback:Dict[str, Any]) -> Dict[str, Any]
```

Deep merge two dicts, preferring values from `preferred`.

## Params
```python
Params(self, params:Dict[str, Any], history:str='', loading_from_archive:bool=False, files_to_archive:Dict[str, str]=None) -> None
```

Represents a parameter dictionary with a history, and contains other functionality around
parameter passing and validation for AllenNLP.

There are currently two benefits of a ``Params`` object over a plain dictionary for parameter
passing:

#. We handle a few kinds of parameter validation, including making sure that parameters
   representing discrete choices actually have acceptable values, and making sure no extra
   parameters are passed.
#. We log all parameter reads, including default values.  This gives a more complete
   specification of the actual parameters used than is given in a JSON file, because
   those may not specify what default values were used, whereas this will log them.

The convention for using a ``Params`` object in AllenNLP is that you will consume the parameters
as you read them, so that there are none left when you've read everything you expect.  This
lets us easily validate that you didn't pass in any `extra` parameters, just by making sure
that the parameter dictionary is empty.  You should do this when you're done handling
parameters, by calling :func:`Params.assert_empty`.

### DEFAULT
The most base type
### add_file_to_archive
```python
Params.add_file_to_archive(self, name:str) -> None
```

Any class in its ``from_params`` method can request that some of its
input files be added to the archive by calling this method.

For example, if some class ``A`` had an ``input_file`` parameter, it could call

```
params.add_file_to_archive("input_file")
```

which would store the supplied value for ``input_file`` at the key
``previous.history.and.then.input_file``. The ``files_to_archive`` dict
is shared with child instances via the ``_check_is_dict`` method, so that
the final mapping can be retrieved from the top-level ``Params`` object.

NOTE: You must call ``add_file_to_archive`` before you ``pop()``
the parameter, because the ``Params`` instance looks up the value
of the filename inside itself.

If the ``loading_from_archive`` flag is True, this will be a no-op.

### pop
```python
Params.pop(self, key:str, default:Any=<object object at 0x107e3bd90>, keep_as_dict:bool=False) -> Any
```

Performs the functionality associated with dict.pop(key), along with checking for
returned dictionaries, replacing them with Param objects with an updated history
(unless keep_as_dict is True, in which case we leave them as dictionaries).

If ``key`` is not present in the dictionary, and no default was specified, we raise a
``ConfigurationError``, instead of the typical ``KeyError``.

### pop_int
```python
Params.pop_int(self, key:str, default:Any=<object object at 0x107e3bd90>) -> int
```

Performs a pop and coerces to an int.

### pop_float
```python
Params.pop_float(self, key:str, default:Any=<object object at 0x107e3bd90>) -> float
```

Performs a pop and coerces to a float.

### pop_bool
```python
Params.pop_bool(self, key:str, default:Any=<object object at 0x107e3bd90>) -> bool
```

Performs a pop and coerces to a bool.

### get
```python
Params.get(self, key:str, default:Any=<object object at 0x107e3bd90>)
```

Performs the functionality associated with dict.get(key) but also checks for returned
dicts and returns a Params object in their place with an updated history.

### pop_choice
```python
Params.pop_choice(self, key:str, choices:List[Any], default_to_first_choice:bool=False, allow_class_names:bool=True) -> Any
```

Gets the value of ``key`` in the ``params`` dictionary, ensuring that the value is one of
the given choices. Note that this `pops` the key from params, modifying the dictionary,
consistent with how parameters are processed in this codebase.

Parameters
----------
key: str
    Key to get the value from in the param dictionary
choices: List[Any]
    A list of valid options for values corresponding to ``key``.  For example, if you're
    specifying the type of encoder to use for some part of your model, the choices might be
    the list of encoder classes we know about and can instantiate.  If the value we find in
    the param dictionary is not in ``choices``, we raise a ``ConfigurationError``, because
    the user specified an invalid value in their parameter file.
default_to_first_choice: bool, optional (default=False)
    If this is ``True``, we allow the ``key`` to not be present in the parameter
    dictionary.  If the key is not present, we will use the return as the value the first
    choice in the ``choices`` list.  If this is ``False``, we raise a
    ``ConfigurationError``, because specifying the ``key`` is required (e.g., you `have` to
    specify your model class when running an experiment, but you can feel free to use
    default settings for encoders if you want).
allow_class_names : bool, optional (default = True)
    If this is `True`, then we allow unknown choices that look like fully-qualified class names.
    This is to allow e.g. specifying a model type as my_library.my_model.MyModel
    and importing it on the fly. Our check for "looks like" is extremely lenient
    and consists of checking that the value contains a '.'.

### as_dict
```python
Params.as_dict(self, quiet:bool=False, infer_type_and_cast:bool=False)
```

Sometimes we need to just represent the parameters as a dict, for instance when we pass
them to PyTorch code.

Parameters
----------
quiet: bool, optional (default = False)
    Whether to log the parameters before returning them as a dict.
infer_type_and_cast : bool, optional (default = False)
    If True, we infer types and cast (e.g. things that look like floats to floats).

### as_flat_dict
```python
Params.as_flat_dict(self)
```

Returns the parameters of a flat dictionary from keys to values.
Nested structure is collapsed with periods.

### duplicate
```python
Params.duplicate(self) -> 'Params'
```

Uses ``copy.deepcopy()`` to create a duplicate (but fully distinct)
copy of these Params.

### assert_empty
```python
Params.assert_empty(self, class_name:str)
```

Raises a ``ConfigurationError`` if ``self.params`` is not empty.  We take ``class_name`` as
an argument so that the error message gives some idea of where an error happened, if there
was one.  ``class_name`` should be the name of the `calling` class, the one that got extra
parameters (if there are any).

### from_file
```python
Params.from_file(params_file:str, params_overrides:str='', ext_vars:dict=None) -> 'Params'
```

Load a `Params` object from a configuration file.

Parameters
----------
params_file : ``str``
    The path to the configuration file to load.
params_overrides : ``str``, optional
    A dict of overrides that can be applied to final object.
    e.g. {"model.embedding_dim": 10}
ext_vars : ``dict``, optional
    Our config files are Jsonnet, which allows specifying external variables
    for later substitution. Typically we substitute these using environment
    variables; however, you can also specify them here, in which case they
    take priority over environment variables.
    e.g. {"HOME_DIR": "/Users/allennlp/home"}

### as_ordered_dict
```python
Params.as_ordered_dict(self, preference_orders:List[List[str]]=None) -> collections.OrderedDict
```

Returns Ordered Dict of Params from list of partial order preferences.

Parameters
----------
preference_orders: List[List[str]], optional
    ``preference_orders`` is list of partial preference orders. ["A", "B", "C"] means
    "A" > "B" > "C". For multiple preference_orders first will be considered first.
    Keys not found, will have last but alphabetical preference. Default Preferences:
    ``[["dataset_reader", "iterator", "model", "train_data_path", "validation_data_path",
    "test_data_path", "trainer", "vocabulary"], ["type"]]``

### get_hash
```python
Params.get_hash(self) -> str
```

Returns a hash code representing the current state of this ``Params`` object.  We don't
want to implement ``__hash__`` because that has deeper python implications (and this is a
mutable object), but this will give you a representation of the current state.
We use `zlib.adler32` instead of Python's builtin `hash` because the random seed for the
latter is reset on each new program invocation, as discussed here:
https://stackoverflow.com/questions/27954892/deterministic-hashing-in-python-3.

## pop_choice
```python
pop_choice(params:Dict[str, Any], key:str, choices:List[Any], default_to_first_choice:bool=False, history:str='?.', allow_class_names:bool=True) -> Any
```

Performs the same function as :func:`Params.pop_choice`, but is required in order to deal with
places that the Params object is not welcome, such as inside Keras layers.  See the docstring
of that method for more detail on how this function works.

This method adds a ``history`` parameter, in the off-chance that you know it, so that we can
reproduce :func:`Params.pop_choice` exactly.  We default to using "?." if you don't know the
history, so you'll have to fix that in the log if you want to actually recover the logged
parameters.

