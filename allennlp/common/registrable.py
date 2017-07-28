from collections import defaultdict
from typing import TypeVar, Type, Dict, List  # pylint: disable=unused-import

from allennlp.common.checks import ConfigurationError

T = TypeVar('T')

class Registrable:
    """
    Any class that inherits from ``Registrable`` gains access to a named registry for its
    subclasses. To register them, just decorate them with the classmethod
    ``@BaseClass.register(name)``.

    After which you can call ``BaseClass.list_available()`` to get the keys for the
    registered subclasses, and ``BaseClass.by_name(name)`` to get the corresponding subclass.
    Note that the registry stores the subclasses themselves; not class instances.
    In most cases you would then call ``from_params(params)`` on the returned subclass.

    You can specify a default by setting ``BaseClass.default_implementation``.
    If it is set, it will be the first element of ``list_available()``.
    """

    _registry = defaultdict(dict)  # type: Dict[Type, Dict[str, Type]]
    default_implementation = None  # type: str

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registrable._registry[cls]
        def add_subclass_to_registry(subclass: Type[T]):
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                message = "Cannot register %s as %s; name already in use for %s" % (
                        name, cls.__name__, registry[name].__name__)
                raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        return Registrable._registry[cls].get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List default first if it exists"""
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        elif default not in keys:
            message = "Default implementation %s is not registered" % default
            raise ConfigurationError(message)
        else:
            return [default] + [k for k in keys if k != default]
