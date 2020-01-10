# allennlp.training.util

Helper functions for Trainers

## HasBeenWarned
```python
HasBeenWarned(self, /, *args, **kwargs)
```

### tqdm_ignores_underscores
bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.
## sparse_clip_norm
```python
sparse_clip_norm(parameters, max_norm, norm_type=2) -> float
```
Clips gradient norm of an iterable of parameters.

The norm is computed over all gradients together, as if they were
concatenated into a single vector. Gradients are modified in-place.
Supports sparse gradients.

Parameters
----------
parameters : ``(Iterable[torch.Tensor])``
    An iterable of Tensors that will have gradients normalized.
max_norm : ``float``
    The max norm of the gradients.
norm_type : ``float``
    The type of the used p-norm. Can be ``'inf'`` for infinity norm.

Returns
-------
Total norm of the parameters (viewed as a single vector).

## move_optimizer_to_cuda
```python
move_optimizer_to_cuda(optimizer)
```

Move the optimizer state to GPU, if necessary.
After calling, any parameter specific state in the optimizer
will be located on the same device as the parameter.

## get_batch_size
```python
get_batch_size(batch:Union[Dict, torch.Tensor]) -> int
```

Returns the size of the batch dimension. Assumes a well-formed batch,
returns 0 otherwise.

## time_to_str
```python
time_to_str(timestamp:int) -> str
```

Convert seconds past Epoch to human readable string.

## str_to_time
```python
str_to_time(time_str:str) -> datetime.datetime
```

Convert human readable string to datetime.datetime.

## datasets_from_params
```python
datasets_from_params(params:allennlp.common.params.Params, cache_directory:str=None, cache_prefix:str=None) -> Dict[str, Iterable[allennlp.data.instance.Instance]]
```

Load all the datasets specified by the config.

Parameters
----------
params : ``Params``
cache_directory : ``str``, optional
    If given, we will instruct the ``DatasetReaders`` that we construct to cache their
    instances in this location (or read their instances from caches in this location, if a
    suitable cache already exists).  This is essentially a `base` directory for the cache, as
    we will additionally add the ``cache_prefix`` to this directory, giving an actual cache
    location of ``cache_directory + cache_prefix``.
cache_prefix : ``str``, optional
    This works in conjunction with the ``cache_directory``.  The idea is that the
    ``cache_directory`` contains caches for all different parameter settings, while the
    ``cache_prefix`` captures a specific set of parameters that led to a particular cache file.
    That is, if you change the tokenization settings inside your ``DatasetReader``, you don't
    want to read cached data that used the old settings.  In order to avoid this, we compute a
    hash of the parameters used to construct each ``DatasetReader`` and use that as a "prefix"
    to the cache files inside the base ``cache_directory``.  So, a given ``input_file`` would
    be cached essentially as ``cache_directory + cache_prefix + input_file``, where you specify
    a ``cache_directory``, the ``cache_prefix`` is based on the dataset reader parameters, and
    the ``input_file`` is whatever path you provided to ``DatasetReader.read()``.  In order to
    allow you to give recognizable names to these prefixes if you want them, you can manually
    specify the ``cache_prefix``.  Note that in some rare cases this can be dangerous, as we'll
    use the `same` prefix for both train and validation dataset readers.

## create_serialization_dir
```python
create_serialization_dir(params:allennlp.common.params.Params, serialization_dir:str, recover:bool, force:bool) -> None
```

This function creates the serialization directory if it doesn't exist.  If it already exists
and is non-empty, then it verifies that we're recovering from a training with an identical configuration.

Parameters
----------
params : ``Params``
    A parameter object specifying an AllenNLP Experiment.
serialization_dir : ``str``
    The directory in which to save results and logs.
recover : ``bool``
    If ``True``, we will try to recover from an existing serialization directory, and crash if
    the directory doesn't exist, or doesn't match the configuration we're given.
force : ``bool``
    If ``True``, we will overwrite the serialization directory if it already exists.

## rescale_gradients
```python
rescale_gradients(model:allennlp.models.model.Model, grad_norm:Union[float, NoneType]=None) -> Union[float, NoneType]
```

Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.

## get_metrics
```python
get_metrics(model:allennlp.models.model.Model, total_loss:float, num_batches:int, reset:bool=False, world_size:int=1, cuda_device:Union[int, List]=0) -> Dict[str, float]
```

Gets the metrics but sets ``"loss"`` to
the total loss divided by the ``num_batches`` so that
the ``"loss"`` metric is "average loss per batch".

