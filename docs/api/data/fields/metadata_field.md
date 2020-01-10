# allennlp.data.fields.metadata_field

## MetadataField
```python
MetadataField(self, metadata:Any) -> None
```

A ``MetadataField`` is a ``Field`` that does not get converted into tensors.  It just carries
side information that might be needed later on, for computing some third-party metric, or
outputting debugging information, or whatever else you need.  We use this in the BiDAF model,
for instance, to keep track of question IDs and passage token offsets, so we can more easily
use the official evaluation script to compute metrics.

We don't try to do any kind of smart combination of this field for batched input - when you use
this ``Field`` in a model, you'll get a list of metadata objects, one for each instance in the
batch.

Parameters
----------
metadata : ``Any``
    Some object containing the metadata that you want to store.  It's likely that you'll want
    this to be a dictionary, but it could be anything you want.

### get_padding_lengths
```python
MetadataField.get_padding_lengths(self) -> Dict[str, int]
```

If there are things in this field that need padding, note them here.  In order to pad a
batch of instance, we get all of the lengths from the batch, take the max, and pad
everything to that length (or use a pre-specified maximum length).  The return value is a
dictionary mapping keys to lengths, like {'num_tokens': 13}.

This is always called after :func:`index`.

### as_tensor
```python
MetadataField.as_tensor(self, padding_lengths:Dict[str, int]) -> ~DataArray
```

Given a set of specified padding lengths, actually pad the data in this field and return a
torch Tensor (or a more complex data structure) of the correct shape.  We also take a
couple of parameters that are important when constructing torch Tensors.

Parameters
----------
padding_lengths : ``Dict[str, int]``
    This dictionary will have the same keys that were produced in
    :func:`get_padding_lengths`.  The values specify the lengths to use when padding each
    relevant dimension, aggregated across all instances in a batch.

### empty_field
```python
MetadataField.empty_field(self) -> 'MetadataField'
```

So that ``ListField`` can pad the number of fields in a list (e.g., the number of answer
option ``TextFields``), we need a representation of an empty field of each type.  This
returns that.  This will only ever be called when we're to the point of calling
:func:`as_tensor`, so you don't need to worry about ``get_padding_lengths``,
``count_vocab_items``, etc., being called on this empty field.

We make this an instance method instead of a static method so that if there is any state
in the Field, we can copy it over (e.g., the token indexers in ``TextField``).

### batch_tensors
```python
MetadataField.batch_tensors(self, tensor_list:List[~DataArray]) -> List[~DataArray]
```

Takes the output of ``Field.as_tensor()`` from a list of ``Instances`` and merges it into
one batched tensor for this ``Field``.  The default implementation here in the base class
handles cases where ``as_tensor`` returns a single torch tensor per instance.  If your
subclass returns something other than this, you need to override this method.

This operation does not modify ``self``, but in some cases we need the information
contained in ``self`` in order to perform the batching, so this is an instance method, not
a class method.

