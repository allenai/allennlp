# allennlp.nn.regularizers.regularizer_applicator

## RegularizerApplicator
```python
RegularizerApplicator(self, regularizers:Sequence[Tuple[str, allennlp.nn.regularizers.regularizer.Regularizer]]=()) -> None
```

Applies regularizers to the parameters of a Module based on regex matches.

### from_params
```python
RegularizerApplicator.from_params(params:Iterable[Tuple[str, allennlp.common.params.Params]]=()) -> Union[_ForwardRef('RegularizerApplicator'), NoneType]
```

Converts a List of pairs (regex, params) into an RegularizerApplicator.
This list should look like

[["regex1", {"type": "l2", "alpha": 0.01}], ["regex2", "l1"]]

where each parameter receives the penalty corresponding to the first regex
that matches its name (which may be no regex and hence no penalty).
The values can either be strings, in which case they correspond to the names
of regularizers, or dictionaries, in which case they must contain the "type"
key, corresponding to the name of a regularizer. In addition, they may contain
auxiliary named parameters which will be fed to the regularizer itself.
To determine valid auxiliary parameters, please refer to the torch.nn.init documentation.

Parameters
----------
params : ``Params``, required.
    A Params object containing a "regularizers" key.

Returns
-------
A RegularizerApplicator containing the specified Regularizers,
or ``None`` if no Regularizers are specified.

