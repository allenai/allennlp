# allennlp.models.archival

Helper functions for archiving models and restoring archived models.

## Archive
```python
Archive(self, /, *args, **kwargs)
```
An archive comprises a Model and its experimental config
### config
Alias for field number 1
### model
Alias for field number 0
### extract_module
```python
Archive.extract_module(self, path:str, freeze:bool=True) -> torch.nn.modules.module.Module
```

This method can be used to load a module from the pretrained model archive.

It is also used implicitly in FromParams based construction. So instead of using standard
params to construct a module, you can instead load a pretrained module from the model
archive directly. For eg, instead of using params like {"type": "module_type", ...}, you
can use the following template::

    {
        "_pretrained": {
            "archive_file": "../path/to/model.tar.gz",
            "path": "path.to.module.in.model",
            "freeze": False
        }
    }

If you use this feature with FromParams, take care of the following caveat: Call to
initializer(self) at end of model initializer can potentially wipe the transferred parameters
by reinitializing them. This can happen if you have setup initializer regex that also
matches parameters of the transferred module. To safe-guard against this, you can either
update your initializer regex to prevent conflicting match or add extra initializer::

    [
        [".*transferred_module_name.*", "prevent"]]
    ]

Parameters
----------
path : ``str``, required
    Path of target module to be loaded from the model.
    Eg. "_textfield_embedder.token_embedder_tokens"
freeze : ``bool``, optional (default=True)
    Whether to freeze the module parameters or not.


## archive_model
```python
archive_model(serialization_dir:str, weights:str='best.th', files_to_archive:Dict[str, str]=None, archive_path:str=None) -> None
```

Archive the model weights, its training configuration, and its
vocabulary to `model.tar.gz`. Include the additional ``files_to_archive``
if provided.

Parameters
----------
serialization_dir : ``str``
    The directory where the weights and vocabulary are written out.
weights : ``str``, optional (default=_DEFAULT_WEIGHTS)
    Which weights file to include in the archive. The default is ``best.th``.
files_to_archive : ``Dict[str, str]``, optional (default=None)
    A mapping {flattened_key -> filename} of supplementary files to include
    in the archive. That is, if you wanted to include ``params['model']['weights']``
    then you would specify the key as `"model.weights"`.
archive_path : ``str``, optional, (default = None)
    A full path to serialize the model to. The default is "model.tar.gz" inside the
    serialization_dir. If you pass a directory here, we'll serialize the model
    to "model.tar.gz" inside the directory.

## load_archive
```python
load_archive(archive_file:str, cuda_device:int=-1, overrides:str='', weights_file:str=None) -> allennlp.models.archival.Archive
```

Instantiates an Archive from an archived `tar.gz` file.

Parameters
----------
archive_file : ``str``
    The archive file to load the model from.
weights_file : ``str``, optional (default = None)
    The weights file to use.  If unspecified, weights.th in the archive_file will be used.
cuda_device : ``int``, optional (default = -1)
    If `cuda_device` is >= 0, the model will be loaded onto the
    corresponding GPU. Otherwise it will be loaded onto the CPU.
overrides : ``str``, optional (default = "")
    JSON overrides to apply to the unarchived ``Params`` object.

