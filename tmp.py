from typing import Any, Dict, Type
import inspect


def get_arg_params(argument: Any, annotation: Type) -> Any:
    existing_params = getattr(argument, '_params', None)
    if existing_params is not None:
        return existing_params
    elif annotation in {float, int}:
        return argument
    else:
        # There are several more cases to handle here if we want to actually do this, but not *that*
        # many more.  We just have to cover all base python types.  Unions might be tricky to figure
        # out, but should still be doable.  This logic basically mirrors what we do when creating
        # objects in FromParams, we're just going the other way.
        raise ValueError()


def get_params(init, *args, **kwargs) -> Dict[str, Any]:
    signature = inspect.signature(init)
    parameters = dict(signature.parameters)
    # need some fancy logic here to match *args and **kwargs with the parametrs.  It's deterministic
    # and doable, but more than I want to write right now.  The below is a quick and dirty first
    # pass.  Oh, hmm, super classes and **kwargs _inside_ the __init__ method make this more tricky,
    # but still doable.
    arg_list = list(args)
    saved_params = {}
    for param_name, param in parameters.items():
        if param_name == "self":
            continue
        if param_name in kwargs:
            argument = kwargs.pop(param_name)
        else:
            argument = arg_list.pop(0)
        argument_params = get_arg_params(argument, param.annotation)
        saved_params[param_name] = argument_params
    return saved_params


class Meta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        new_cls = super(Meta, mcs).__new__(mcs, name, bases, namespace, **kwargs)
        user_init = new_cls.__init__
        def __init__(self, *args, **kwargs):
            self._params = get_params(user_init, *args, **kwargs)
            user_init(self, *args, **kwargs)
        setattr(new_cls, '__init__', __init__)
        return new_cls


class FromParams(metaclass=Meta):
    pass


class Gaussian(FromParams):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance


class NestedGaussian(FromParams):
    def __init__(self, gaussian: Gaussian, alpha: float):
        self.gaussian = gaussian
        self.alpha = alpha


g = Gaussian(1.3, 2.3)
print(g._params)
# {'mean': 1.3, 'variance': 2.3}
g = Gaussian(mean=1.8, variance=4.3)
print(g._params)
# {'mean': 1.8, 'variance': 4.3}

n = NestedGaussian(Gaussian(mean=1.8, variance=4.3), alpha=.01)
print(n._params)
# {'gaussian': {'mean': 1.8, 'variance': 4.3}, 'alpha': 0.01}
