from typing import TypeVar, Type, Dict, Union, Any, cast
import inspect
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T = TypeVar('T')

# getattr is necessary because mypy insists that no such attribute exists.
_NO_DEFAULT = getattr(inspect, '_empty')  # pylint: disable=invalid-name

def takes_arg(obj, arg: str) -> bool:
    """
    Checks whether the provided obj takes a certain arg.
    If it's a class, we're really checking whether its constructor does.
    If it's a function or method, we're checking the object itself.
    Otherwise, we raise an error.
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f"object {obj} is not callable")
    return arg in signature.parameters

def remove_optional(annotation):
    """
    Optional[X] annotations are actually represented as Union[X, NoneType].
    For our purposes, the "Optional" part is not interesting, so here we
    throw it away.
    """
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())
    if origin == Union and len(args) == 2 and args[1] == type(None):
        return args[0]
    else:
        return annotation

def create_kwargs(cls: Type[T], params: Params, **extras) -> Dict[str, Any]:
    """
    Given some class, a `Params` object, and potentially other keyword arguments,
    create a dict of keyword args suitable for passing to the class's constructor.
    """
    # Get the signature of the constructor.
    signature = inspect.signature(cls.__init__)
    kwargs: Dict[str, Any] = {}

    # Iterate over all the constructor parameters and their annotations.
    for name, param in signature.parameters.items():
        # Skip "self". You're not *required* to call the first parameter "self",
        # so in theory this logic is fragile, but if you don't call the self parameter
        # "self" you kind of deserve what happens.
        if name == "self":
            continue

        # Get the annotation and break it into `origin` and `args`
        annotation = remove_optional(param.annotation)
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', [])
        default = param.default
        optional = default != _NO_DEFAULT

        # Some constructors expect extra non-parameter items, e.g. vocab: Vocabulary.
        # We check the provided `extras` for these and just use them if they exist.
        if name in extras:
            kwargs[name] = extras[name]

        # The next case is when the parameter type is itself constructible from_params.
        elif hasattr(annotation, 'from_params'):
            if name in params:
                # Our params have an entry for this, so we use that.
                subparams = params.pop(name)

                if takes_arg(annotation.from_params, 'extras'):
                    # If annotation.params accepts **extras, we need to pass them all along.
                    subextras = extras
                else:
                    # Otherwise, only supply the ones that are actual args; any additional ones
                    # will cause a TypeError.
                    subextras = {k: v for k, v in extras.items() if takes_arg(annotation.from_params, k)}

                # In some cases we allow a string instead of a param dict, so
                # we need to handle that case separately.
                if isinstance(subparams, str):
                    kwargs[name] = annotation.by_name(subparams)()
                else:
                    kwargs[name] = annotation.from_params(params=subparams, **subextras)
            elif not optional:
                # Not optional and not supplied, that's an error!
                raise ConfigurationError(f"expected key {name} for {cls.__name__}")

        # If the parameter type is a Python primitive, just pop it off
        # using the correct casting pop_xyz operation.
        elif annotation == str:
            kwargs[name] = (params.pop(name, default)
                            if optional
                            else params.pop(name))
        elif annotation == int:
            kwargs[name] = (params.pop_int(name, default)
                            if optional
                            else params.pop_int(name))
        elif annotation == bool:
            kwargs[name] = (params.pop_bool(name, default)
                            if optional
                            else params.pop_bool(name))
        elif annotation == float:
            kwargs[name] = (params.pop_float(name, default)
                            if optional
                            else params.pop_float(name))

        # This handles types like Dict[str, TokenIndexer], where we need to instantiate
        # each value from_params and return the resulting dict.
        elif origin == Dict and len(args) == 2 and hasattr(args[-1], 'from_params'):
            value_cls = annotation.__args__[-1]

            value_dict = {}

            for key, value_params in params.pop(name, Params({})).items():
                value_dict[key] = value_cls.from_params(params=value_params, **extras)

            kwargs[name] = value_dict

        else:
            # Pass it on as is and hope for the best.   ¯\_(ツ)_/¯
            if optional:
                kwargs[name] = params.pop(name, default)
            else:
                kwargs[name] = params.pop(name)

    params.assert_empty(cls.__name__)
    return {k: v for k, v in kwargs.items() if takes_arg(cls, k)}



class FromParams:
    """
    Mixin to give a from_params method to classes. We create a distinct base class for this
    because sometimes we want non-Registrable classes to be instantiatable from_params.
    """
    @classmethod
    def from_params(cls: Type[T], params: Params, **extras) -> T:
        """
        This is the automatic implementation of `from_params`. Any class that subclasses `FromParams`
        (or `Registrable`, which itself subclasses `FromParams`) gets this implementation for free.
        If you want your class to be instantiated from params in the "obvious" way -- pop off parameters
        and hand them to your constructor with the same names -- this provides that functionality.

        If you need more complex logic in your from `from_params` method, you'll have to implement
        your own method that overrides this one.
        """
        # pylint: disable=protected-access
        from allennlp.common.registrable import Registrable  # import here to avoid circular imports

        logger.info(f"instantiating class {cls} from params {getattr(params, 'params', params)} "
                    f"and extras {extras}")

        if params is None:
            return None

        registered_subclasses = Registrable._registry.get(cls)

        if registered_subclasses is not None:
            # We know ``cls`` inherits from Registrable, so we'll use a cast to make mypy happy.
            # We have to use a disable to make pylint happy.
            # pylint: disable=no-member
            as_registrable = cast(Type[Registrable], cls)
            default_to_first_choice = as_registrable.default_implementation is not None
            choice = params.pop_choice("type",
                                       choices=as_registrable.list_available(),
                                       default_to_first_choice=default_to_first_choice)
            subclass = registered_subclasses[choice]

            # Prune down the dict of extras, if necessary.
            if not takes_arg(subclass.from_params, 'extras'):
                extras = {k: v for k, v in extras.items() if takes_arg(subclass.from_params, k)}

            return subclass.from_params(params=params, **extras)
        else:
            # This is not a base class, so convert our params and extras into a dict of kwargs.
            kwargs = create_kwargs(cls, params, **extras)
            return cls(**kwargs)  # type: ignore
