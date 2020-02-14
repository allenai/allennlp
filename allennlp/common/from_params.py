"""
One of the design principles of AllenNLP is the use of a modular,
declarative language (JSON) for defining experiments and models.

This is implemented by giving each AllenNLP class a method

.. code-block
    @classmethod
    def from_params(cls, params: Params, **extras) -> 'ClassName':
        ...

that contains the logic for instantiating a class instance from a JSON-like
`Params` object. Historically you had to implement your own `from_params`
method on every class you wanted to instantiate this way, even though
most of the time you were simply popping off params and handing them to the
constructor (making sure that you popped them using the same default values
as in the constructor.)

It turns out that in those simple cases, we can generate a `from_params`
method automatically. This implementation lives in the `FromParams` class.
Every `Registrable` subclass automatically gets it, and you can have your
non-`Registrable` classes subclass from it as well.

The inclusion of `extras` allows for non-FromParams parameters to be passed
as well. For instance, all of our `Model` subclasses require a
`Vocabulary` parameter. Accordingly, the `train` command calls

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

from copy import deepcopy
from pathlib import Path
from typing import TypeVar, Type, Callable, Dict, Union, Any, cast, List, Tuple, Set
import inspect
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.lazy import Lazy
from allennlp.common.params import Params

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="FromParams")

# If a function parameter has no default value specified,
# this is what the inspect module returns.
_NO_DEFAULT = inspect.Parameter.empty


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


def takes_kwargs(obj) -> bool:
    """
    Checks whether a provided object takes in any positional arguments.
    Similar to takes_arg, we do this for both the __init__ function of
    the class or a function / method
    Otherwise, we raise an error
    """
    if inspect.isclass(obj):
        signature = inspect.signature(obj.__init__)
    elif inspect.ismethod(obj) or inspect.isfunction(obj):
        signature = inspect.signature(obj)
    else:
        raise ConfigurationError(f"object {obj} is not callable")
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD  # type: ignore
        for p in signature.parameters.values()
    )


def can_construct_from_params(type_: Type) -> bool:
    if type_ in [str, int, float, bool]:
        return True
    origin = getattr(type_, "__origin__", None)
    if origin == Lazy:
        return True
    elif origin:
        if hasattr(type_, "from_params"):
            return True
        args = getattr(type_, "__args__")
        return all(can_construct_from_params(arg) for arg in args)
    return hasattr(type_, "from_params")


def remove_optional(annotation: type):
    """
    Optional[X] annotations are actually represented as Union[X, NoneType].
    For our purposes, the "Optional" part is not interesting, so here we
    throw it away.
    """
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    if origin == Union and len(args) == 2 and args[1] == type(None):  # noqa
        return args[0]
    else:
        return annotation


def infer_params(cls: Type[T], constructor: Callable[..., T] = None):
    if constructor is None:
        constructor = cls.__init__

    signature = inspect.signature(constructor)
    parameters = dict(signature.parameters)

    has_kwargs = False
    for param in parameters.values():
        if param.kind == param.VAR_KEYWORD:
            has_kwargs = True

    if not has_kwargs:
        return parameters

    # "mro" is "method resolution order".  The first one is the current class, the next is the
    # first superclass, and so on.  Taking the first superclass should work in almost all cases that
    # we're looking for here.  This could fail, though, if you are using multiple inheritance and we
    # pick the wrong superclass.  We'll worry about how to fix that when we run into an actual
    # problem because of it.
    super_class = cls.mro()[1]
    super_parameters = infer_params(super_class)

    return {**super_parameters, **parameters}  # Subclass parameters overwrite superclass ones


def create_kwargs(
    constructor: Callable[..., T], cls: Type[T], params: Params, **extras
) -> Dict[str, Any]:
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

    kwargs: Dict[str, Any] = {}

    parameters = infer_params(cls, constructor)

    # Iterate over all the constructor parameters and their annotations.
    for param_name, param in parameters.items():
        # Skip "self". You're not *required* to call the first parameter "self",
        # so in theory this logic is fragile, but if you don't call the self parameter
        # "self" you kind of deserve what happens.
        if param_name == "self":
            continue
        # Also skip **kwargs parameters; we handled them above.
        if param.kind == param.VAR_KEYWORD:
            continue

        # If the annotation is a compound type like typing.Dict[str, int],
        # it will have an __origin__ field indicating `typing.Dict`
        # and an __args__ field indicating `(str, int)`. We capture both.
        annotation = remove_optional(param.annotation)

        constructed_arg = pop_and_construct_arg(
            cls.__name__, param_name, annotation, param.default, params, **extras
        )

        # If we just ended up constructing the default value for the parameter, we can just omit it.
        # Leaving it in can cause issues with **kwargs in some corner cases, where you might end up
        # with multiple values for a single parameter (e.g., the default value gives you lazy=False
        # for a dataset reader inside **kwargs, but a particular dataset reader actually hard-codes
        # lazy=True - the superclass sees both lazy=True and lazy=False in its constructor).
        if constructed_arg is not param.default:
            kwargs[param_name] = constructed_arg

    params.assert_empty(cls.__name__)
    return kwargs


def create_extras(cls: Type[T], extras: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a dictionary of extra arguments, returns a dictionary of
    kwargs that actually are a part of the signature of the cls.from_params
    (or cls) method.
    """
    subextras: Dict[str, Any] = {}
    if hasattr(cls, "from_params"):
        from_params_method = cls.from_params  # type: ignore
    else:
        # In some rare cases, we get a registered subclass that does _not_ have a
        # from_params method (this happens with Activations, for instance, where we
        # register pytorch modules directly).  This is a bit of a hack to make those work,
        # instead of adding a `from_params` method for them somehow. Then the extras
        # in the class constructor are what we are looking for, to pass on.
        from_params_method = cls
    if takes_kwargs(from_params_method):
        # If annotation.params accepts **kwargs, we need to pass them all along.
        # For example, `BasicTextFieldEmbedder.from_params` requires a Vocabulary
        # object, but `TextFieldEmbedder.from_params` does not.
        subextras = extras
    else:
        # Otherwise, only supply the ones that are actual args; any additional ones
        # will cause a TypeError.
        subextras = {k: v for k, v in extras.items() if takes_arg(from_params_method, k)}
    return subextras


def pop_and_construct_arg(
    class_name: str, argument_name: str, annotation: Type, default: Any, params: Params, **extras
) -> Any:
    """
    Does the work of actually constructing an individual argument for
    [`create_kwargs`](./from_params#create_kwargs).

    Here we're in the inner loop of iterating over the parameters to a particular constructor,
    trying to construct just one of them.  The information we get for that parameter is its name,
    its type annotation, and its default value; we also get the full set of `Params` for
    constructing the object (which we may mutate), and any `extras` that the constructor might
    need.

    We take the type annotation and default value here separately, instead of using an
    `inspect.Parameter` object directly, so that we can handle `Union` types using recursion on
    this method, trying the different annotation types in the union in turn.
    """
    from allennlp.models.archival import load_archive  # import here to avoid circular imports

    # We used `argument_name` as the method argument to avoid conflicts with 'name' being a key in
    # `extras`, which isn't _that_ unlikely.  Now that we are inside the method, we can switch back
    # to using `name`.
    name = argument_name

    # Some constructors expect extra non-parameter items, e.g. vocab: Vocabulary.
    # We check the provided `extras` for these and just use them if they exist.
    if name in extras:
        if name not in params:
            return extras[name]
        else:
            logger.warning(
                f"Parameter {name} for class {class_name} was found in both "
                "**extras and in params. Using the specification found in params, "
                "but you probably put a key in a config file that you didn't need, "
                "and if it is different from what we get from **extras, you might "
                "get unexpected behavior."
            )
    # Next case is when argument should be loaded from pretrained archive.
    elif (
        name in params
        and isinstance(params.get(name), Params)
        and "_pretrained" in params.get(name)
    ):
        load_module_params = params.pop(name).pop("_pretrained")
        archive_file = load_module_params.pop("archive_file")
        module_path = load_module_params.pop("module_path")
        freeze = load_module_params.pop("freeze", True)
        archive = load_archive(archive_file)
        result = archive.extract_module(module_path, freeze)
        if not isinstance(result, annotation):
            raise ConfigurationError(
                f"The module from model at {archive_file} at path {module_path} "
                f"was expected of type {annotation} but is of type {type(result)}"
            )
        return result

    popped_params = params.pop(name, default) if default != _NO_DEFAULT else params.pop(name)
    if popped_params is None:
        origin = getattr(annotation, "__origin__", None)
        if origin == Lazy:
            return Lazy(lambda **kwargs: None)
        return None

    return construct_arg(class_name, name, popped_params, annotation, default, **extras)


def construct_arg(
    class_name: str,
    argument_name: str,
    popped_params: Params,
    annotation: Type,
    default: Any,
    **extras,
) -> Any:
    """
    The first two parameters here are only used for logging if we encounter an error.
    """
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", [])

    # The parameter is optional if its default value is not the "no default" sentinel.
    optional = default != _NO_DEFAULT

    if hasattr(annotation, "from_params"):
        if popped_params is default:
            return default
        elif popped_params is not None:
            # Our params have an entry for this, so we use that.

            subextras = create_extras(annotation, extras)

            # In some cases we allow a string instead of a param dict, so
            # we need to handle that case separately.
            if isinstance(popped_params, str):
                return annotation.by_name(popped_params)()
            else:
                if isinstance(popped_params, dict):
                    popped_params = Params(popped_params)
                return annotation.from_params(params=popped_params, **subextras)
        elif not optional:
            # Not optional and not supplied, that's an error!
            raise ConfigurationError(f"expected key {argument_name} for {class_name}")
        else:
            return default

    # If the parameter type is a Python primitive, just pop it off
    # using the correct casting pop_xyz operation.
    elif annotation in {int, bool}:
        if type(popped_params) in {int, bool}:
            return annotation(popped_params)
        else:
            raise TypeError(f"Expected {argument_name} to be a {annotation.__name__}.")
    elif annotation == str:
        # Strings are special because we allow casting from Path to str.
        if type(popped_params) == str or isinstance(popped_params, Path):
            return str(popped_params)  # type: ignore
        else:
            raise TypeError(f"Expected {argument_name} to be a string.")
    elif annotation == float:
        # Floats are special because in Python, you can put an int wherever you can put a float.
        # https://mypy.readthedocs.io/en/stable/duck_type_compatibility.html
        if type(popped_params) in {int, float}:
            return popped_params
        else:
            raise TypeError(f"Expected {argument_name} to be numeric.")

    # This is special logic for handling types like Dict[str, TokenIndexer],
    # List[TokenIndexer], Tuple[TokenIndexer, Tokenizer], and Set[TokenIndexer],
    # which it creates by instantiating each value from_params and returning the resulting structure.
    elif origin in (Dict, dict) and len(args) == 2 and can_construct_from_params(args[-1]):
        value_cls = annotation.__args__[-1]

        value_dict = {}

        for key, value_params in popped_params.items():
            value_dict[key] = construct_arg(
                str(value_cls),
                argument_name + "." + key,
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )

        return value_dict

    elif origin in (List, list) and len(args) == 1 and can_construct_from_params(args[0]):
        value_cls = annotation.__args__[0]

        value_list = []

        for i, value_params in enumerate(popped_params):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_list.append(value)

        return value_list

    elif origin in (Tuple, tuple) and all(can_construct_from_params(arg) for arg in args):
        value_list = []

        for i, (value_cls, value_params) in enumerate(zip(annotation.__args__, popped_params)):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_list.append(value)

        return tuple(value_list)

    elif origin in (Set, set) and len(args) == 1 and can_construct_from_params(args[0]):
        value_cls = annotation.__args__[0]

        value_set = set()

        for i, value_params in enumerate(popped_params):
            value = construct_arg(
                str(value_cls),
                argument_name + f".{i}",
                value_params,
                value_cls,
                _NO_DEFAULT,
                **extras,
            )
            value_set.add(value)

        return value_set

    elif origin == Union:
        # Storing this so we can recover it later if we need to.
        backup_params = deepcopy(popped_params)

        # We'll try each of the given types in the union sequentially, returning the first one that
        # succeeds.
        for arg_annotation in args:
            try:
                return construct_arg(
                    str(arg_annotation),
                    argument_name,
                    popped_params,
                    arg_annotation,
                    default,
                    **extras,
                )
            except (ValueError, TypeError, ConfigurationError, AttributeError):
                # Our attempt to construct the argument may have modified popped_params, so we
                # restore it here.
                popped_params = deepcopy(backup_params)

        # If none of them succeeded, we crash.
        raise ConfigurationError(
            f"Failed to construct argument {argument_name} with type {annotation}"
        )
    elif origin == Lazy:
        if popped_params is default:
            return Lazy(lambda **kwargs: default)
        value_cls = args[0]
        subextras = create_extras(value_cls, extras)

        def constructor(**kwargs):
            # If there are duplicate keys between subextras and kwargs, this will overwrite the ones
            # in subextras with what's in kwargs.  If an argument shows up twice, we should take it
            # from what's passed to Lazy.construct() instead of what we got from create_extras().
            # Almost certainly these will be identical objects, anyway.
            subextras.update(kwargs)
            return value_cls.from_params(params=popped_params, **subextras)

        return Lazy(constructor)  # type: ignore
    else:
        # Pass it on as is and hope for the best.   ¯\_(ツ)_/¯
        if isinstance(popped_params, Params):
            return popped_params.as_dict(quiet=True)
        return popped_params


class FromParams:
    """
    Mixin to give a from_params method to classes. We create a distinct base class for this
    because sometimes we want non-Registrable classes to be instantiatable from_params.
    """

    @classmethod
    def from_params(
        cls: Type[T],
        params: Params,
        constructor_to_call: Callable[..., T] = None,
        constructor_to_inspect: Callable[..., T] = None,
        **extras,
    ) -> T:
        """
        This is the automatic implementation of `from_params`. Any class that subclasses
        `FromParams` (or `Registrable`, which itself subclasses `FromParams`) gets this
        implementation for free.  If you want your class to be instantiated from params in the
        "obvious" way -- pop off parameters and hand them to your constructor with the same names --
        this provides that functionality.

        If you need more complex logic in your from `from_params` method, you'll have to implement
        your own method that overrides this one.

        The `constructor_to_call` and `constructor_to_inspect` arguments deal with a bit of
        redirection that we do.  We allow you to register particular `@classmethods` on a class as
        the constructor to use for a registered name.  This lets you, e.g., have a single
        `Vocabulary` class that can be constructed in two different ways, with different names
        registered to each constructor.  In order to handle this, we need to know not just the class
        we're trying to construct (`cls`), but also what method we should inspect to find its
        arguments (`constructor_to_inspect`), and what method to call when we're done constructing
        arguments (`constructor_to_call`).  These two methods are the same when you've used a
        `@classmethod` as your constructor, but they are `different` when you use the default
        constructor (because you inspect `__init__`, but call `cls()`).
        """

        from allennlp.common.registrable import Registrable  # import here to avoid circular imports

        logger.info(
            f"instantiating class {cls} from params {getattr(params, 'params', params)} "
            f"and extras {set(extras.keys())}"
        )

        if params is None:
            return None

        if isinstance(params, str):
            params = Params({"type": params})

        registered_subclasses = Registrable._registry.get(cls)

        if registered_subclasses is not None and not constructor_to_call:
            # We know `cls` inherits from Registrable, so we'll use a cast to make mypy happy.

            as_registrable = cast(Type[Registrable], cls)
            default_to_first_choice = as_registrable.default_implementation is not None
            choice = params.pop_choice(
                "type",
                choices=as_registrable.list_available(),
                default_to_first_choice=default_to_first_choice,
            )
            subclass, constructor_name = as_registrable.resolve_class_name(choice)
            # See the docstring for an explanation of what's going on here.
            if not constructor_name:
                constructor_to_inspect = subclass.__init__
                constructor_to_call = subclass  # type: ignore
            else:
                constructor_to_inspect = getattr(subclass, constructor_name)
                constructor_to_call = constructor_to_inspect

            if hasattr(subclass, "from_params"):
                # We want to call subclass.from_params.
                extras = create_extras(subclass, extras)
                # mypy can't follow the typing redirection that we do, so we explicitly cast here.
                retyped_subclass = cast(Type[T], subclass)
                return retyped_subclass.from_params(
                    params=params,
                    constructor_to_call=constructor_to_call,
                    constructor_to_inspect=constructor_to_inspect,
                    **extras,
                )
            else:
                # In some rare cases, we get a registered subclass that does _not_ have a
                # from_params method (this happens with Activations, for instance, where we
                # register pytorch modules directly).  This is a bit of a hack to make those work,
                # instead of adding a `from_params` method for them somehow.  We just trust that
                # you've done the right thing in passing your parameters, and nothing else needs to
                # be recursively constructed.
                extras = create_extras(subclass, extras)
                constructor_args = {**params, **extras}
                return subclass(**constructor_args)  # type: ignore
        else:
            # This is not a base class, so convert our params and extras into a dict of kwargs.

            # See the docstring for an explanation of what's going on here.
            if not constructor_to_inspect:
                constructor_to_inspect = cls.__init__
            if not constructor_to_call:
                constructor_to_call = cls

            if constructor_to_inspect == object.__init__:
                # This class does not have an explicit constructor, so don't give it any kwargs.
                # Without this logic, create_kwargs will look at object.__init__ and see that
                # it takes *args and **kwargs and look for those.
                kwargs: Dict[str, Any] = {}
            else:
                # This class has a constructor, so create kwargs for it.
                kwargs = create_kwargs(constructor_to_inspect, cls, params, **extras)

            return constructor_to_call(**kwargs)  # type: ignore
