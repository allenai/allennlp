"""
One of the design principles of AllenNLP is the use of a modular,
declarative language (JSON) for defining experiments and models.

This is implemented by giving each AllenNLP class a method

.. code-block
    @classmethod
    def from_params(cls, params: Params, **extras) -> 'ClassName':
        ...

that contains the logic for instantiating a class instance from a JSON-like
``Params`` object. Historically you had to implement your own ``from_params``
method on every class you wanted to instantiate this way, even though
most of the time you were simply popping off params and handing them to the
constructor (making sure that you popped them using the same default values
as in the constructor.)

It turns out that in those simple cases, we can generate a ``from_params``
method automatically. This implementation lives in the ``FromParams`` class.
Every ``Registrable`` subclass automatically gets it, and you can have your
non-``Registrable`` classes subclass from it as well.

The inclusion of ``extras`` allows for non-FromParams parameters to be passed
as well. For instance, all of our ``Model`` subclasses require a
``Vocabulary`` parameter. Accordingly, the ``train`` command calls

```
model = Model.from_params(params=params.pop('model'), vocab=vocab)
```

As an AllenNLP user, you will probably never need to worry about this.
However, if you do, note that the extra arguments must be called by keyword.
Prior to this default implementation it was possible to call them positionally
but this is no longer the case.

In some cases you might want the construction of class instances `from_params`
to include more elaborate logic than "pop off params and hand them to the constructor".
In this case your class just needs to explicitly implement its own `from_params`
method.
"""

from typing import TypeVar, Type, Dict, Union, Any, cast
import inspect
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T = TypeVar('T')

# If a function parameter has no default value specified,
# this is what the inspect module returns.
_NO_DEFAULT = inspect.Parameter.empty  # pylint: disable=invalid-name

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

def remove_optional(annotation: type):
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

    The function does this by finding the class's constructor, matching the constructor
    arguments to entries in the `params` object, and instantiating values for the parameters
    using the type annotation and possibly a from_params method.

    Any values that are provided in the `extras` will just be used as is.
    For instance, you might provide an existing `Vocabulary` this way.
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

        # If the annotation is a compound type like typing.Dict[str, int],
        # it will have an __origin__ field indicating `typing.Dict`
        # and an __args__ field indicating `(str, int)`. We capture both.
        annotation = remove_optional(param.annotation)
        origin = getattr(annotation, '__origin__', None)
        args = getattr(annotation, '__args__', [])

        # The parameter is optional if its default value is not the "no default" sentinel.
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
                    # For example, `BasicTextFieldEmbedder.from_params` requires a Vocabulary
                    # object, but `TextFieldEmbedder.from_params` does not.
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
                    print(annotation)
                    kwargs[name] = annotation.from_params(params=subparams, **subextras)
            elif not optional:
                # Not optional and not supplied, that's an error!
                raise ConfigurationError(f"expected key {name} for {cls.__name__}")
            else:
                kwargs[name] = default

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

        # This is special logic for handling types like Dict[str, TokenIndexer], which it creates by
        # instantiating each value from_params and returning the resulting dict.
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
    return kwargs


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

            # We want to call subclass.from_params. It's possible that it's just the "free"
            # implementation here, in which case it accepts `**extras` and we are not able
            # to make any assumptions about what extra parameters it needs.
            #
            # It's also possible that it has a custom `from_params` method. In that case it
            # won't accept any **extra parameters and we'll need to filter them out.
            if not takes_arg(subclass.from_params, 'extras'):
                # Necessarily subclass.from_params is a custom implementation, so we need to
                # pass it only the args it's expecting.
                extras = {k: v for k, v in extras.items() if takes_arg(subclass.from_params, k)}

            return subclass.from_params(params=params, **extras)
        else:
            # This is not a base class, so convert our params and extras into a dict of kwargs.

            if cls.__init__ == object.__init__:
                # This class does not have an explicit constructor, so don't give it any kwargs.
                # Without this logic, create_kwargs will look at object.__init__ and see that
                # it takes *args and **kwargs and look for those.
                kwargs: Dict[str, Any] = {}
            else:
                # This class has a constructor, so create kwargs for it.
                kwargs = create_kwargs(cls, params, **extras)

            return cls(**kwargs)  # type: ignore
