import inspect
from functools import wraps

_FLAGS = {'STORE_FUNCTION_RESULTS': False}

def store_function_results(flag: bool) -> None:
    _FLAGS['STORE_FUNCTION_RESULTS'] = flag

def store_result(func):
    @wraps(func)
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)

        if _FLAGS['STORE_FUNCTION_RESULTS']:
            # Import here to avoid circularity
            from allennlp.models.model import Model

            stack = inspect.stack()

            calling_module = stack[1].frame.f_locals['self']

            if isinstance(calling_module, Model):
                secret_functions = getattr(calling_module, '_stored_function_results', [])
                secret_functions.append([func.__name__, result])
                setattr(calling_module, '_stored_function_results', secret_functions)

        return result

    return inner
