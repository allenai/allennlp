from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Lazy(Generic[T]):
    """
    This class is for use when constructing objects using `FromParams`, when an argument to a
    constructor has a _sequential dependency_ with another argument to the same constructor.  For
    example, in a `Trainer` class you might want to take a `Model` and an `Optimizer` as arguments,
    but the `Optimizer` needs to be constructed using the parameters from the `Model`.  You can give
    the type annotation `Lazy[Optimizer]` to the optimizer argument, then inside the constructor
    call `optimizer.construct(parameters=model.parameters)`.

    This is only recommended for use when you have registered a `@classmethod` as the constructor
    for your class, instead of using `__init__`.  Having a `Lazy[]` type annotation on an argument
    to an `__init__` method makes your class completely dependent on being constructed using the
    `FromParams` pipeline, which is not a good idea.

    The actual implementation here is incredibly simple; the logic that handles the lazy
    construction is actually found in `FromParams`, where we have a special case for a `Lazy` type
    annotation.
    """

    def __init__(self, constructor: Callable[..., T]):
        self._constructor = constructor

    def construct(self, **kwargs) -> T:
        return self._constructor(**kwargs)
