"""
If a class (e.g. a ``Model`` or a ``DatasetReader``) inherits from ``Archivable``
then we can ``collect()`` a dictionary of arbitrary data to be serialized
from the class instance itself as well as recursively from any member elements
(including ``List`` and ``Dict`` values) that themselves implement ``Archivable``.

Conversely, we can use ``populate_from_collection()``

Note that this chain must be unbroken. That is, if we had

```
class A(Archivable):
    b: B

class B:
    c: C

class C(Archivable):
    pass
```

then calling ``add_to_archives`` on an instance of ``A`` would not archive the ``C``
contained in the intermediate ``B`` (whereas it would if ``B`` implemented ``Archivable``).
"""
from typing import Any, Iterable, Tuple, Dict
import inspect
import shelve

class Archivable:
    """
    Any class that inherits from ``Archivable`` gains archiving methods.
    """
    def stuff_to_archive(self) -> Dict[str, Any]:
        """
        A class that implements ``Archivable`` should override this method which simply
        returns a ``dict`` {name -> value} of things it wants archived.

        During de-archiving, it will receive an identical dict with the same names.
        """
        return {}

    def populate_stuff_from_archive(self, stuff: Dict[str, Any]) -> None:
        """
        A class that implements ``Archivable`` should override this method which takes
        the ``dict`` of "unarchived" items and uses them to restore some state.
        """
        pass

    def collect(self, prefix: str = '') -> Dict[str, Any]:
        return {name: obj.stuff_to_archive()
                for name, obj in _archivables(self, prefix)}

    def populate_from_collection(self, collection: Dict[str, Any], prefix: str) -> None:
        for name, obj in _archivables(self, prefix):
            obj.populate_stuff_from_archive(collection[name])


def _archivables(instance: Any, prefix: str = '') -> Iterable[Tuple[str, Archivable]]:
    """
    Given an object and a prefix, recursively explores and yields
    all ``Archivable`` members along with unique "paths" to them. It will explore
    list elements, dict values, and instances of classes that themselves implement
    ``Archivable``.
    """
    # Start by yielding this instance
    if isinstance(instance, Archivable):
        yield prefix, instance

    # ``dir`` gets way too many properties, so filter intelligently.
    for name in dir(instance):
        # Retrieve the property with this name
        prop = getattr(instance, name)
        # Skip __ methods and the registry
        if name.startswith("__") or name == "_registry":
            continue
        # Skip functions and methods
        if any(check(prop) for check in [inspect.isfunction, inspect.ismethod]):
            continue
        # For a dictionary, explore all the values
        if isinstance(prop, dict):
            for k, v in prop.items():
                yield from _archivables(v, f"{prefix}.{name}.{k}")
        # For a list, explore all the items
        elif isinstance(prop, list):
            for i, v in enumerate(prop):
                yield from _archivables(v, f"{prefix}.{name}.{i}")
        # For an instance of ``Archivable``, make a recursive call.
        elif isinstance(prop, Archivable):
            yield from _archivables(prop, f"{prefix}.{name}")
