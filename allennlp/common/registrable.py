"""
:class:`~allennlp.common.registrable.Registrable` is a "mixin" for endowing
any base class with a named registry for its subclasses and a decorator
for registering them.
"""
from collections import defaultdict
from typing import TypeVar, Type, Dict, List
import importlib
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.from_params import FromParams

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T = TypeVar('T')

class Registrable(FromParams):
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
    def register(cls: Type[T], name: str, exist_ok=False):
        """
        Register a class under a particular name.

        Parameters
        ----------
        name: ``str``
            The name to register the class under.
        exist_ok: ``bool`, optional (default=False)
            If True, overwrites any existing models registered under ``name``. Else,
            throws an error if a model is already registered under ``name``.
        """
        registry = Registrable._registry[cls]
        def add_subclass_to_registry(subclass: Type[T]):
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                if exist_ok:
                    message = (f"{name} has already been registered as {registry[name].__name__}, but "
                               f"exist_ok=True, so overwriting with {cls.__name__}")
                    logger.info(message)
                else:
                    message = (f"Cannot register {name} as {cls.__name__}; "
                               f"name already in use for {registry[name].__name__}")
                    raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        logger.info(f"instantiating registered subclass {name} of {cls}")
        if name in Registrable._registry[cls]:
            return Registrable._registry[cls].get(name)
        elif "." in name:
            # This might be a fully qualified class name, so we'll try importing its "module"
            # and finding it there.
            parts = name.split(".")
            submodule = ".".join(parts[:-1])
            class_name = parts[-1]

            try:
                module = importlib.import_module(submodule)
            except ModuleNotFoundError:
                raise ConfigurationError(f"tried to interpret {name} as a path to a class "
                                         f"but unable to import module {submodule}")

            try:
                return getattr(module, class_name)
            except AttributeError:
                raise ConfigurationError(f"tried to interpret {name} as a path to a class "
                                         f"but unable to find class {class_name} in {submodule}")

        else:
            # is not a qualified class name
            raise ConfigurationError(f"{name} is not a registered name for {cls.__name__}. "
                                     "You probably need to use the --include-package flag "
                                     "to load your custom code. Alternatively, you can specify your choices "
                                     """using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} """
                                     "in which case they will be automatically imported correctly.")


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
