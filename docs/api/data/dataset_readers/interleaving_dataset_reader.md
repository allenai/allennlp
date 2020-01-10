# allennlp.data.dataset_readers.interleaving_dataset_reader

## InterleavingDatasetReader
```python
InterleavingDatasetReader(self, readers:Dict[str, allennlp.data.dataset_readers.dataset_reader.DatasetReader], dataset_field_name:str='dataset', scheme:str='round_robin', lazy:bool=False) -> None
```

A ``DatasetReader`` that wraps multiple other dataset readers,
and interleaves their instances, adding a ``MetadataField`` to
indicate the provenance of each instance.

Unlike most of our other dataset readers, here the ``file_path`` passed into
``read()`` should be a JSON-serialized dictionary with one file_path
per wrapped dataset reader (and with corresponding keys).

Parameters
----------
readers : ``Dict[str, DatasetReader]``
    The dataset readers to wrap. The keys of this dictionary will be used
    as the values in the MetadataField indicating provenance.
dataset_field_name : str, optional (default = "dataset")
    The name of the MetadataField indicating which dataset an instance came from.
scheme : str, optional (default = "round_robin")
    Indicates how to interleave instances. Currently the two options are "round_robin",
    which repeatedly cycles through the datasets grabbing one instance from each;
    and "all_at_once", which yields all the instances from the first dataset,
    then all the instances from the second dataset, and so on. You could imagine also
    implementing some sort of over- or under-sampling, although hasn't been done.
lazy : bool, optional (default = False)
    If this is true, ``instances()`` will return an object whose ``__iter__`` method
    reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.

