# allennlp.training.checkpointer

## Checkpointer
```python
Checkpointer(self, serialization_dir:str=None, keep_serialized_model_every_num_seconds:int=None, num_serialized_models_to_keep:int=20) -> None
```

This class implements the functionality for checkpointing your model and trainer state
during training. It is agnostic as to what those states look like (they are typed as
Dict[str, Any]), but they will be fed to ``torch.save`` so they should be serializable
in that sense. They will also be restored as Dict[str, Any], which means the calling
code is responsible for knowing what to do with them.

### find_latest_checkpoint
```python
Checkpointer.find_latest_checkpoint(self) -> Tuple[str, str]
```

Return the location of the latest model and training state files.
If there isn't a valid checkpoint then return None.

### restore_checkpoint
```python
Checkpointer.restore_checkpoint(self) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

Restores a model from a serialization_dir to the last saved checkpoint.
This includes a training state (typically consisting of an epoch count and optimizer state),
which is serialized separately from  model parameters. This function should only be used to
continue training - if you wish to load a model for inference/load parts of a model into a new
computation graph, you should use the native Pytorch functions:
`` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
this function will do nothing and return empty dicts.

Returns
-------
states: Tuple[Dict[str, Any], Dict[str, Any]]
    The model state and the training state.

