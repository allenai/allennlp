# allennlp.common.checks

Functions and exceptions for checking that
AllenNLP and its models are configured correctly.

## ConfigurationError
```python
ConfigurationError(self, message)
```

The exception raised by any AllenNLP object when it's misconfigured
(e.g. missing properties, invalid properties, unknown properties).

## ExperimentalFeatureWarning
```python
ExperimentalFeatureWarning(self, /, *args, **kwargs)
```

A warning that you are using an experimental feature
that may change or be deleted.

## parse_cuda_device
```python
parse_cuda_device(cuda_device:Union[str, int, List[int]]) -> int
```

Disambiguates single GPU and multiple GPU settings for cuda_device param.

