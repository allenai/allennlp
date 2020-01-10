# allennlp.training.callbacks.checkpoint

## Checkpoint
```python
Checkpoint(self, checkpointer:allennlp.training.checkpointer.Checkpointer, model_save_interval:Union[float, NoneType]=None, state_dict_attrs:List[str]=None, other_attrs:List[str]=None) -> None
```

Callback that orchestrates checkpointing of your model and training state.

Parameters
----------
checkpointer : ``Checkpointer``
    The checkpoint reader and writer to use.
model_save_interval : ``float``, optional (default=None)
    If provided, then serialize models every ``model_save_interval``
    seconds within single epochs.  In all cases, models are also saved
    at the end of every epoch if ``serialization_dir`` is provided.
state_dict_attrs : ``List[str]``, optional (default = ['optimizer'])
    The attributes of the Trainer state whose `.state_dict()`
    should be persisted at each checkpoint.
other_attrs : ``List[str]``, optional (default = ['batch_num_total'])
    The attributes of the Trainer state that should be persisted
    as-is at each checkpoint.

