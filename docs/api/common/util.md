# allennlp.common.util

Various utilities that don't fit anwhere else.

## sanitize
```python
sanitize(x:Any) -> Any
```

Sanitize turns PyTorch and Numpy types into basic Python types so they
can be serialized into JSON.

## group_by_count
```python
group_by_count(iterable:List[Any], count:int, default_value:Any) -> List[List[Any]]
```

Takes a list and groups it into sublists of size ``count``, using ``default_value`` to pad the
list at the end if the list is not divisable by ``count``.

For example:
>>> group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 0)
[[1, 2, 3], [4, 5, 6], [7, 0, 0]]

This is a short method, but it's complicated and hard to remember as a one-liner, so we just
make a function out of it.

## lazy_groups_of
```python
lazy_groups_of(iterable:Iterable[~A], group_size:int) -> Iterator[List[~A]]
```

Takes an iterable and batches the individual instances into lists of the
specified size. The last list may be smaller if there are instances left over.

## pad_sequence_to_length
```python
pad_sequence_to_length(sequence:List, desired_length:int, default_value:Callable[[], Any]=<function <lambda> at 0x126330620>, padding_on_right:bool=True) -> List
```

Take a list of objects and pads it to the desired length, returning the padded list.  The
original list is not modified.

Parameters
----------
sequence : List
    A list of objects to be padded.

desired_length : int
    Maximum length of each sequence. Longer sequences are truncated to this length, and
    shorter ones are padded to it.

default_value: Callable, default=lambda: 0
    Callable that outputs a default value (of any type) to use as padding values.  This is
    a lambda to avoid using the same object when the default value is more complex, like a
    list.

padding_on_right : bool, default=True
    When we add padding tokens (or truncate the sequence), should we do it on the right or
    the left?

Returns
-------
padded_sequence : List

## add_noise_to_dict_values
```python
add_noise_to_dict_values(dictionary:Dict[~A, float], noise_param:float) -> Dict[~A, float]
```

Returns a new dictionary with noise added to every key in ``dictionary``.  The noise is
uniformly distributed within ``noise_param`` percent of the value for every value in the
dictionary.

## namespace_match
```python
namespace_match(pattern:str, namespace:str)
```

Matches a namespace pattern against a namespace string.  For example, ``*tags`` matches
``passage_tags`` and ``question_tags`` and ``tokens`` matches ``tokens`` but not
``stemmed_tokens``.

## prepare_environment
```python
prepare_environment(params:allennlp.common.params.Params)
```

Sets random seeds for reproducible experiments. This may not work as expected
if you use this from within a python project in which you have already imported Pytorch.
If you use the scripts/run_model.py entry point to training models with this library,
your experiments should be reasonably reproducible. If you are using this from your own
project, you will want to call this function before importing Pytorch. Complete determinism
is very difficult to achieve with libraries doing optimized linear algebra due to massively
parallel execution, which is exacerbated by using GPUs.

Parameters
----------
params: Params object or dict, required.
    A ``Params`` object or dict holding the json parameters.

## FileFriendlyLogFilter
```python
FileFriendlyLogFilter(self, name='')
```

TQDM and requests use carriage returns to get the training line to update for each batch
without adding more lines to the terminal output.  Displaying those in a file won't work
correctly, so we'll just make sure that each batch shows up on its one line.

## get_spacy_model
```python
get_spacy_model(spacy_model_name:str, pos_tags:bool, parse:bool, ner:bool) -> spacy.language.Language
```

In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
keyed by the options we used to create the spacy model, so any particular configuration only
gets loaded once.

## import_submodules
```python
import_submodules(package_name:str) -> None
```

Import all submodules under the given package.
Primarily useful so that people using AllenNLP as a library
can specify their own custom packages and have their custom
classes get loaded and registered.

## peak_memory_mb
```python
peak_memory_mb() -> float
```

Get peak memory usage for this process, as measured by
max-resident-set size:

https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size

Only works on OSX and Linux, returns 0.0 otherwise.

## gpu_memory_mb
```python
gpu_memory_mb() -> Dict[int, int]
```

Get the current GPU memory usage.
Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

Returns
-------
``Dict[int, int]``
    Keys are device ids as integers.
    Values are memory usage as integers in MB.
    Returns an empty ``dict`` if GPUs are not available.

## ensure_list
```python
ensure_list(iterable:Iterable[~A]) -> List[~A]
```

An Iterable may be a list or a generator.
This ensures we get a list without making an unnecessary copy.

## is_lazy
```python
is_lazy(iterable:Iterable[~A]) -> bool
```

Checks if the given iterable is lazy,
which here just means it's not a list.

## is_master
```python
is_master(rank:int=None, world_size:int=None) -> bool
```

Checks if the process is a "master" in a distributed process group. If a
process group is not initialized, this returns `True`.

Parameters
----------
rank : int ( default = None )
    Global rank of the process if in a distributed process group. If not
    given, rank is obtained using `torch.distributed.get_rank()`
world_size : int ( default = None )
    Number of processes in the distributed group. If not
    given, this is obtained using `torch.distributed.get_world_size()`

## is_distributed
```python
is_distributed() -> bool
```

Checks if the distributed process group is available and has been initialized

## sanitize_wordpiece
```python
sanitize_wordpiece(wordpiece:str) -> str
```

Sanitizes wordpieces from BERT or RoBERTa tokenizers.

