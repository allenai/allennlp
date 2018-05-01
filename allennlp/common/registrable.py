"""
:class:`~allennlp.common.registrable.Registrable` is a "mixin" for endowing
any base class with a named registry for its subclasses and a decorator
for registering them.
"""

from collections import defaultdict
from typing import TypeVar, Type, Dict, List, Any, Callable
import inspect

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params

T = TypeVar('T')

_DEFAULT_TO_FIRST_CHOICE = {
      'TextFieldEmbedder',
}

def takes_arg(f: Callable, arg: str) -> bool:
    signature = inspect.signature(f)
    print("takes_arg", f, arg, signature, signature.parameters)
    return arg in signature.parameters

def create_kwargs(cls: Type[T], params: Params, **extras) -> Dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    print(signature)
    kwargs: Dict[str, Any] = {}

    for name, param in signature.parameters.items():
        print("create_kwargs", cls, name, param)
        # Don't need to sub in for `self`
        if name == "self":
            continue

        annotation = param.annotation
        default = param.default

        if name in extras:
            # This is for stuff like Vocabulary, which is passed in manually
            kwargs[name] = extras[name]

        elif hasattr(annotation, 'from_params'):
            if name in params:
                subparams = params.pop(name)

                if not takes_arg(annotation.from_params, 'extras'):
                    subextras = {k:v for k, v in extras.items() if takes_arg(annotation.from_params, k)}
                else:
                    subextras = extras

                kwargs[name] = annotation.from_params(params=subparams, **subextras)
            elif default == inspect._empty:
                raise ConfigurationError(f"expected key {name} for {cls.__name__}")

        elif annotation == str:
            kwargs[name] = (params.pop(name, default)
                            if default != inspect._empty
                            else params.pop(name))
        elif annotation == int:
            kwargs[name] = (params.pop_int(name, default)
                            if default != inspect._empty
                            else params.pop_int(name))
        elif annotation == bool:
            kwargs[name] = (params.pop_bool(name, default)
                            if default != inspect._empty
                            else params.pop_bool(name))
        elif annotation == float:
            kwargs[name] = (params.pop_float(name, default)
                            if default != inspect._empty
                            else params.pop_float(name))

        elif (getattr(annotation, '__origin__', None) == Dict and
              hasattr(annotation, '__args__') and
              hasattr(annotation.__args__[-1], 'from_params')):
            value_cls = annotation.__args__[-1]

            value_dict = {}

            for key, value_params in params.pop(name, Params({})).items():
                value_dict[key] = value_cls.from_params(params=value_params, **extras)

            kwargs[name] = value_dict

    #params.assert_empty(cls.__name__)
    return {k: v for k, v in kwargs.items() if takes_arg(cls, k)}

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

    Note that if you use this class to implement a new ``Registrable`` abstract class,
    you must ensure that all subclasses of the abstract class are loaded when the module is
    loaded, because the subclasses register themselves in their respective files. You can
    achieve this by having the abstract class and all subclasses in the __init__.py of the
    module in which they reside (as this causes any import of either the abstract class or
    a subclass to load all other subclasses and the abstract class).
    """

    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None

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
        if name not in Registrable._registry[cls]:
            raise ConfigurationError("%s is not a registered name for %s" % (name, cls.__name__))
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

    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) -> T:
        # pylint: disable=protected-access
        if params is None:
            return None

        # If this is the base class, delegate to the subclass.
        print("from_params", cls, params, extras)
        if "type" in params or cls.__name__ in _DEFAULT_TO_FIRST_CHOICE:
            choice = params.pop_choice("type",
                                       cls.list_available(),
                                       cls.__name__ in _DEFAULT_TO_FIRST_CHOICE)
            subclass = cls.by_name(choice)
            if not takes_arg(subclass.from_params, 'extras'):
                extras = {k:v for k, v in extras.items() if takes_arg(subclass.from_params, k)}
            return subclass.from_params(params=params, **extras)
        else:
            kwargs = create_kwargs(cls, params, **extras)
            return cls(**kwargs)
