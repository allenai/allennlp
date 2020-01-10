# allennlp.common.from_params

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

## takes_arg
```python
takes_arg(obj, arg:str) -> bool
```

Checks whether the provided obj takes a certain arg.
If it's a class, we're really checking whether its constructor does.
If it's a function or method, we're checking the object itself.
Otherwise, we raise an error.

## takes_kwargs
```python
takes_kwargs(obj) -> bool
```

Checks whether a provided object takes in any positional arguments.
Similar to takes_arg, we do this for both the __init__ function of
the class or a function / method
Otherwise, we raise an error

## remove_optional
```python
remove_optional(annotation:type)
```

Optional[X] annotations are actually represented as Union[X, NoneType].
For our purposes, the "Optional" part is not interesting, so here we
throw it away.

## create_kwargs
```python
create_kwargs(cls:Type[~T], params:allennlp.common.params.Params, **extras) -> Dict[str, Any]
```

Given some class, a `Params` object, and potentially other keyword arguments,
create a dict of keyword args suitable for passing to the class's constructor.

The function does this by finding the class's constructor, matching the constructor
arguments to entries in the `params` object, and instantiating values for the parameters
using the type annotation and possibly a from_params method.

Any values that are provided in the `extras` will just be used as is.
For instance, you might provide an existing `Vocabulary` this way.

## create_extras
```python
create_extras(cls:Type[~T], extras:Dict[str, Any]) -> Dict[str, Any]
```

Given a dictionary of extra arguments, returns a dictionary of
kwargs that actually are a part of the signature of the cls.from_params
(or cls) method.

## construct_arg
```python
construct_arg(cls:Type[~T], param_name:str, annotation:Type, default:Any, params:allennlp.common.params.Params, **extras) -> Any
```

Does the work of actually constructing an individual argument for :func:`create_kwargs`.

Here we're in the inner loop of iterating over the parameters to a particular constructor,
trying to construct just one of them.  The information we get for that parameter is its name,
its type annotation, and its default value; we also get the full set of ``Params`` for
constructing the object (which we may mutate), and any ``extras`` that the constructor might
need.

We take the type annotation and default value here separately, instead of using an
``inspect.Parameter`` object directly, so that we can handle ``Union`` types using recursion on
this method, trying the different annotation types in the union in turn.

## FromParams
```python
FromParams(self, /, *args, **kwargs)
```

Mixin to give a from_params method to classes. We create a distinct base class for this
because sometimes we want non-Registrable classes to be instantiatable from_params.

### from_params
```python
FromParams.from_params(params:allennlp.common.params.Params, **extras) -> ~T
```

This is the automatic implementation of `from_params`. Any class that subclasses `FromParams`
(or `Registrable`, which itself subclasses `FromParams`) gets this implementation for free.
If you want your class to be instantiated from params in the "obvious" way -- pop off parameters
and hand them to your constructor with the same names -- this provides that functionality.

If you need more complex logic in your from `from_params` method, you'll have to implement
your own method that overrides this one.

