# allennlp.common.tqdm

:class:`~allennlp.common.tqdm.Tqdm` wraps tqdm so we can add configurable
global defaults for certain tqdm parameters.

## Tqdm
```python
Tqdm(self, /, *args, **kwargs)
```

### default_mininterval
float(x) -> floating point number

Convert a string or number to a floating point number, if possible.
### set_slower_interval
```python
Tqdm.set_slower_interval(use_slower_interval:bool) -> None
```

If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
but it is not great for log files.  You might want to set this if you are primarily going
to be looking at output through log files, not the terminal.

