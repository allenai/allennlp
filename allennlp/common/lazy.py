from typing import Callable, Generic, TypeVar, Optional

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

    !!! Warning
        The way this class is used in from_params means that optional constructor arguments CANNOT
        be compared to `None` _before_ it is constructed. See the example below for correct usage.

    ```
    @classmethod
    def my_constructor(cls, some_object: Lazy[MyObject] = None) -> MyClass:
        ...
        # WRONG! some_object will never be None at this point, it will be
        # a Lazy[] that returns None
        obj = some_object or MyObjectDefault()
        # CORRECT:
        obj = some_object.construct(kwarg=kwarg) or MyObjectDefault()
        ...
    ```

    """

    def __init__(self, constructor: Callable[..., T]):
        self._constructor = constructor

    def construct(self, **kwargs) -> Optional[T]:
        return self._constructor(**kwargs)
