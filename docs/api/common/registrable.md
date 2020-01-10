# allennlp.common.registrable

:class:`~allennlp.common.registrable.Registrable` is a "mixin" for endowing
any base class with a named registry for its subclasses and a decorator
for registering them.

## Registrable
```python
Registrable(self, /, *args, **kwargs)
```

Any class that inherits from ``Registrable`` gains access to a named registry for its
subclasses. To register them, just decorate them with the classmethod
``@BaseClass.register(name)``.

After which you can call ``BaseClass.list_available()`` to get the keys for the
registered subclasses, and ``BaseClass.by_name(name)`` to get the corresponding subclass.
Note that the registry stores the subclasses themselves; not class instances.
In most cases you would then call ``from_params(params)`` on the returned subclass.

You can specify a default by setting ``BaseClass.default_implementation``.
If it is set, it will be the first element of ``list_available()``.

Note that if you use this class to implement a new ``Registrable`` abstract class,
you must ensure that all subclasses of the abstract class are loaded when the module is
loaded, because the subclasses register themselves in their respective files. You can
achieve this by having the abstract class and all subclasses in the __init__.py of the
module in which they reside (as this causes any import of either the abstract class or
a subclass to load all other subclasses and the abstract class).

### register
```python
Registrable.register(name:str, exist_ok=False)
```

Register a class under a particular name.

Parameters
----------
name : ``str``
    The name to register the class under.
exist_ok : ``bool``, optional (default=False)
    If True, overwrites any existing models registered under ``name``. Else,
    throws an error if a model is already registered under ``name``.

### list_available
```python
Registrable.list_available() -> List[str]
```
List default first if it exists
