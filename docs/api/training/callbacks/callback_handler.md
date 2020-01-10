# allennlp.training.callbacks.callback_handler

## EventHandler
```python
EventHandler(self, /, *args, **kwargs)
```
EventHandler(name, callback, handler, priority)
### callback
Alias for field number 1
### handler
Alias for field number 2
### name
Alias for field number 0
### priority
Alias for field number 3
## CallbackHandler
```python
CallbackHandler(self, callbacks:Iterable[allennlp.training.callbacks.callback.Callback], state:allennlp.training.trainer_base.TrainerBase, verbose:bool=False) -> None
```

A ``CallbackHandler`` owns zero or more ``Callback``s, each of which is associated
with some "event". It then exposes a ``fire_event`` method, which calls each
callback associated with that event ordered by their priorities.

The callbacks take no parameters; instead they read from and write to this handler's
``state``, which should be a Trainer.

Parameters
----------
callbacks : ``Iterable[Callback]``
    The callbacks to be handled.
state : ``TrainerBase``
    The trainer from which the callbacks will read state
    and to which the callbacks will write state.
verbose : bool, optional (default = False)
    If true, will log every event -> callback. Please only
    use this for debugging purposes.

### callbacks
```python
CallbackHandler.callbacks(self) -> List[allennlp.training.callbacks.callback.Callback]
```

Returns the callbacks associated with this handler.
Each callback may be registered under multiple events,
but we make sure to only return it once. If `typ` is specified,
only returns callbacks of that type.

### fire_event
```python
CallbackHandler.fire_event(self, event:str) -> None
```

Runs every callback registered for the provided event,
ordered by their priorities.

