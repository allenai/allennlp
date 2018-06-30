"""
:class:`~allennlp.common.registrable.Registrable` is a "mixin" for endowing
any base class with a named registry for its subclasses and a decorator
for registering them.
"""

from collections import defaultdict
from typing import TypeVar, Type, Dict, List, Any, Callable, Union
import inspect
import importlib
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params

logger = logging.getLogger(__name__)

T = TypeVar('T')

_DEFAULT_TO_FIRST_CHOICE = {
        'TextFieldEmbedder',
        'WordFilter',
        'WordSplitter',
        'WordStemmer',
        'TokenIndexer',
        'SimilarityFunction'
}

NO_DEFAULT = inspect._empty

def takes_arg(func: Callable, arg: str) -> bool:
    signature = inspect.signature(func)
    #print("takes_arg", func, arg, signature, signature.parameters)
    return arg in signature.parameters

def remove_optional(annotation):
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin == Union and len(args) == 2 and args[1] == type(None):
        return args[0]
    else:
        return annotation


def create_kwargs(cls: Type[T], params: Params, **extras) -> Dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    #print(signature)
    kwargs: Dict[str, Any] = {}

    for name, param in signature.parameters.items():
        #print("create_kwargs", cls, name, param)
        # Don't need to sub in for `self`
        if name == "self":
            continue

        annotation = remove_optional(param.annotation)
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
            elif default == NO_DEFAULT:
                raise ConfigurationError(f"expected key {name} for {cls.__name__}")

        elif annotation == str:
            kwargs[name] = (params.pop(name, default)
                            if default != NO_DEFAULT
                            else params.pop(name))
        elif annotation == int:
            kwargs[name] = (params.pop_int(name, default)
                            if default != NO_DEFAULT
                            else params.pop_int(name))
        elif annotation == bool:
            kwargs[name] = (params.pop_bool(name, default)
                            if default != NO_DEFAULT
                            else params.pop_bool(name))
        elif annotation == float:
            kwargs[name] = (params.pop_float(name, default)
                            if default != NO_DEFAULT
                            else params.pop_float(name))

        elif (getattr(annotation, '__origin__', None) == Dict and
              hasattr(annotation, '__args__') and
              hasattr(annotation.__args__[-1], 'from_params')):
            value_cls = annotation.__args__[-1]

            value_dict = {}

            for key, value_params in params.pop(name, Params({})).items():
                value_dict[key] = value_cls.from_params(params=value_params, **extras)

            kwargs[name] = value_dict

        else:
            # Pass it on as is and hope for the best.   ¯\_(ツ)_/¯
            if default == NO_DEFAULT:
                kwargs[name] = params.pop(name)
            else:
                kwargs[name] = params.pop(name, default)

    #params.assert_empty(cls.__name__)
    return {k: v for k, v in kwargs.items() if takes_arg(cls, k)}

def _load_module(cls: type) -> None:
    module_name = cls.__module__
    #parent = module_name[:module_name.rfind('.')]
    importlib.import_module(module_name)


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
        logger.info(f"instantiating registered subclass {name} of {cls}")
        if name not in Registrable._registry[cls]:
            raise ConfigurationError("%s is not a registered name for %s" % (name, cls.__name__))
        return Registrable._registry[cls].get(name)

    @classmethod
    def list_available(cls) -> List[str]:
        """List default first if it exists"""
        # This is necessary because some of the subclasses may not have been loaded
        # at this point, and they're not registered until they're loaded.
        _load_module(cls)

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
        logger.info(f"instantiating class {cls} from params {params.params} and extras {extras}")
        # pylint: disable=protected-access
        if params is None:
            return None

        # If this is the base class, delegate to the subclass.
        registered_subclasses = Registrable._registry.get(cls)
        #print()
        #print("cls", cls)
        if registered_subclasses is not None:
            default = getattr(cls, 'default_implementation', None)
            if default is not None:
                choices = [default] + [k for k in registered_subclasses if k != default]
            else:
                choices = list(registered_subclasses)

            choice = params.pop_choice("type",
                                       choices=choices,
                                       default_to_first_choice=cls.__name__ in _DEFAULT_TO_FIRST_CHOICE)
            subclass = registered_subclasses[choice]
            if not takes_arg(subclass.from_params, 'extras'):
                extras = {k:v for k, v in extras.items() if takes_arg(subclass.from_params, k)}
            #print("params", params.params)
            #print("extras", extras)
            return subclass.from_params(params=params, **extras)
        else:
            kwargs = create_kwargs(cls, params, **extras)
            #print("kwargs", kwargs)
            return cls(**kwargs)
