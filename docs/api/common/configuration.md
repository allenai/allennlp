# allennlp.common.configuration

Tools for programmatically generating config files for AllenNLP models.

## BASE_CONFIG

A ``Config`` represents an entire subdict in a configuration file.
If it corresponds to a named subclass of a registrable class,
it will also contain a ``type`` item in addition to whatever
items are required by the subclass ``from_params`` method.

### items
list() -> new empty list
list(iterable) -> new list initialized from iterable's items
## VOCAB_CONFIG

A ``Config`` represents an entire subdict in a configuration file.
If it corresponds to a named subclass of a registrable class,
it will also contain a ``type`` item in addition to whatever
items are required by the subclass ``from_params`` method.

### items
list() -> new empty list
list(iterable) -> new list initialized from iterable's items
## full_name
```python
full_name(cla55:Union[type, NoneType]) -> str
```

Return the full name (including module) of the given class.

## ConfigItem
```python
ConfigItem(self, /, *args, **kwargs)
```

Each ``ConfigItem`` represents a single entry in a configuration JsonDict.

### annotation
Alias for field number 1
### comment
Alias for field number 3
### default_value
Alias for field number 2
### name
Alias for field number 0
## Config
```python
Config(self, items:List[allennlp.common.configuration.ConfigItem], typ3:str=None) -> None
```

A ``Config`` represents an entire subdict in a configuration file.
If it corresponds to a named subclass of a registrable class,
it will also contain a ``type`` item in addition to whatever
items are required by the subclass ``from_params`` method.

## render_config
```python
render_config(config:allennlp.common.configuration.Config, indent:str='') -> str
```

Pretty-print a config in sort-of-JSON+comments.

