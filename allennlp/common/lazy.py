import copy
import inspect
from typing import Callable, Generic, TypeVar, Type, Union, Optional, Dict, Any

from allennlp.common.params import Params


T = TypeVar("T")


class Lazy(Generic[T]):
    """
    This class is for use when constructing objects using `FromParams`, when an argument to a
    constructor has a _sequential dependency_ with another argument to the same constructor.

    For example, in a `Trainer` class you might want to take a `Model` and an `Optimizer` as arguments,
    but the `Optimizer` needs to be constructed using the parameters from the `Model`. You can give
    the type annotation `Lazy[Optimizer]` to the optimizer argument, then inside the constructor
    call `optimizer.construct(parameters=model.parameters)`.

    This is only recommended for use when you have registered a `@classmethod` as the constructor
    for your class, instead of using `__init__`.  Having a `Lazy[]` type annotation on an argument
    to an `__init__` method makes your class completely dependent on being constructed using the
    `FromParams` pipeline, which is not a good idea.

    The actual implementation here is incredibly simple; the logic that handles the lazy
    construction is actually found in `FromParams`, where we have a special case for a `Lazy` type
    annotation.

    ```python
    @classmethod
    def my_constructor(
        cls,
        some_object: Lazy[MyObject],
        optional_object: Lazy[MyObject] = None,
        # or:
        #  optional_object: Optional[Lazy[MyObject]] = None,
        optional_object_with_default: Optional[Lazy[MyObject]] = Lazy(MyObjectDefault),
        required_object_with_default: Lazy[MyObject] = Lazy(MyObjectDefault),
    ) -> MyClass:
        obj1 = some_object.construct()
        obj2 = None if optional_object is None else optional_object.construct()
        obj3 = None optional_object_with_default is None else optional_object_with_default.construct()
        obj4 = required_object_with_default.construct()
    ```

    """

    def __init__(
        self,
        constructor: Union[Type[T], Callable[..., T]],
        params: Optional[Params] = None,
        contructor_extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._constructor = constructor
        self._params = params or Params({})
        self._constructor_extras = contructor_extras or {}

    @property
    def constructor(self) -> Callable[..., T]:
        if inspect.isclass(self._constructor):

            def constructor_to_use(**kwargs):
                return self._constructor.from_params(  # type: ignore[union-attr]
                    copy.deepcopy(self._params),
                    **kwargs,
                )

            return constructor_to_use
        else:
            return self._constructor

    def construct(self, **kwargs) -> T:
        """
        Call the constructor to create an instance of `T`.
        """
        # If there are duplicate keys between self._constructor_extras and kwargs,
        # this will overwrite the ones in self._constructor_extras with what's in kwargs.
        contructor_kwargs = {**self._constructor_extras, **kwargs}
        return self.constructor(**contructor_kwargs)
