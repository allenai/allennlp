from collections import OrderedDict
import inspect
import importlib
import json
import pkgutil
from types import ModuleType
from typing import NamedTuple, Dict, Sequence, Callable, Optional, TypeVar, List, Union, Sequence, Tuple, Any

import allennlp
from allennlp.common import JsonDict
from allennlp.common.registrable import Registrable

DEFAULT = object()

class RegisteredClass(NamedTuple):
    module: str
    name: str
    under_registrable: bool
    signature: inspect.Signature
    clas: type

    @staticmethod
    def from_spec(path: str, spec) -> 'RegisteredClass':
        parts = path.split('.')
        module = '.'.join(parts[:-1])
        name = parts[-1]

        parameters = []
        for row in spec:
            if len(row) == 2:
                # no default
                name, annotation = row
                parameters.append(inspect.Parameter(name=name,
                                                    annotation=annotation,
                                                    kind=inspect.Parameter.KEYWORD_ONLY))
            elif len(row) == 3:
                name, annotation, default = row
                parameters.append(inspect.Parameter(name=name,
                                                    annotation=annotation,
                                                    default=default,
                                                    kind=inspect.Parameter.KEYWORD_ONLY))

        signature = inspect.Signature(parameters=parameters)

        return RegisteredClass(module=module, name=name, under_registrable=False, signature=signature, clas=None)

PYTORCH_SIGS = {
    'torch.nn.modules.rnn.LSTM': [
        ('input_size', int,),
        ('hidden_size', int,),
        ('num_layers', int,),
        ('bias', bool, True),
        ('batch_first', bool, True),
        ('dropout', float, 0.0),
        ('bidirectional', bool, False)
    ],
    'torch.nn.modules.rnn.GRU': [
        ('input_size', int,),
        ('hidden_size', int,),
        ('num_layers', int,),
        ('bias', bool, True),
        ('batch_first', bool, True),
        ('dropout', float, 0.0),
        ('bidirectional', bool, False)
    ],
    'torch.nn.modules.rnn.RNN': [
        ('input_size', int,),
        ('hidden_size', int,),
        ('num_layers', int,),
        ('nonlinearity', str, 'relu'),
        ('bias', bool, True),
        ('batch_first', bool, True),
        ('dropout', float, 0.0),
        ('bidirectional', bool, False)
    ],
}

def import_all_submodules(module_name: str):
    """
    Import all submodules of the specified module
    """
    module = importlib.import_module(module_name)

    for _loader, name, _is_pkg in pkgutil.walk_packages(module.__path__):
        try:
            importlib.import_module(f"{module_name}.{name}")
        except ModuleNotFoundError:
            print(f"unable to import {module_name}.{name}, continuing")


def collect_classes(module: ModuleType,
                    result: Dict[type, RegisteredClass] = None) -> Dict[type, RegisteredClass]:
    """
    Collect all the classes underneath the specified module.
    Returns a Dict where the keys are the full qualified name
    of the class, and the values are ``RegisteredClass`` instances.
    """
    result = {} if result is None else result
    name = module.__name__

    for _, obj in inspect.getmembers(module):
        # If this is neither a class or a module, continue.
        if not inspect.isclass(obj) and not inspect.ismodule(obj):
            continue
        try:
            obj_module = obj.__module__
            obj_name = obj.__name__
            key = f"{obj_module}.{obj_name}"
        except AttributeError:
            key = obj.__name__

        if not key.startswith(name):
            continue

        # Find classes that are in this module and haven't been seen yet.
        if (inspect.isclass(obj) and
                    #'from_params' in obj.__dict__ and
                    obj not in result):
            under_registrable = False
            for base in obj.__bases__:
                if base.__name__ == "Registrable":
                    under_registrable = True
                    break
            try:
                signature = inspect.signature(obj.__init__)
            except ValueError:
                print(f"unable to get signature for {obj}, continuing")
                signature = None
            reg = RegisteredClass(obj.__module__, obj.__name__, under_registrable, signature, obj)
            # TODO: don't do this twice
            result[obj] = reg
            result[key] = reg

        if inspect.ismodule(obj):
            if key.startswith(name):
                result = collect_classes(obj, result)

    return result

def full_path(clas: type) -> str:
    try:
        return f"{clas.__module__}.{clas.__name__}"
    except AttributeError:
        # Hack for Pytorch wrappers
        clas = clas._module_class
        return f"{clas.__module__}.{clas.__name__}"

T = TypeVar('T')

def read(prompt: str = '',
         optional: bool = False,
         choices: Sequence[str] = None,
         converter: Callable[[str], T] = str) -> Optional[T]:
    """
    Attempts to read in a value with the given prompt.
    If the value is optional, accepts an empty string as the default.
    If ``choices`` are provided, the value must be one of the choices.
    If you know the value is (say) an ``int`` you can provide ``int``
    as the ``converter`` and it will be applied to the received string.
    """
    while True:
        value = input(prompt + "\n")
        if not value and optional:
            return DEFAULT
        elif not value:
            print("value is required!")
        elif choices and value not in choices:
            print(f"please choose one of {choices}")
        else:
            try:
                return converter(value)
            except ValueError:
                print("not a valid value")

def convert_bool(s: str) -> bool:
    """
    The builtin ``bool`` doesn't do the right thing.
    """
    if s.lower() in ('true', 't'):
        return True
    elif s.lower() in ('false', 'f'):
        return False
    else:
        raise ValueError("please input true or false")


def configure_registrable(class_name: str, path: str) -> JsonDict:
    # Get the list of valid types
    print(path, class_name)
    registered_class = REGISTERED_CLASSES[class_name]
    clas = registered_class.clas
    types = list(REGISTRY[clas].keys())

    if clas.default_implementation:
        default_str = "(default: " + clas.default_implementation + " with all of its default params)"
    else:
        default_str = ""
    prompt = f"please choose a type. {default_str} valid choices {types} : "
    typ = read(prompt, choices=types, optional=clas.default_implementation)

    if typ == DEFAULT:
        return DEFAULT

    subclass = REGISTRY[clas][typ]
    subclass_name = full_path(subclass)
    config = configure(subclass_name, path)
    config['type'] = typ
    return config

BUILTINS = {
    'builtins.int': int,
    'builtins.float': float,
    'builtins.str': str,
    'builtins.bool': convert_bool
}

PROBABLY_SKIP = {
    'vocab',
    'initializer',
    'regularizer'
}

def specify_optional(desc: str) -> bool:
    yes_no = read(f"{desc} is optional, do you want to specify it? type YES if so",
                  optional=True)
    return yes_no and yes_no != DEFAULT and yes_no.lower() == "yes"

def configure(clas: Any,
              path: str,
              prompt: str = '',
              optional: bool = False,
              skip_keys: Sequence[str] = ()):
    print(path)

    # Deal with primitive types first:
    if clas in (int, float, str):
        return read(prompt, optional=optional, converter=clas)
    elif clas == bool:
        return read(prompt, optional=optional, converter=convert_bool)

    # Check for optional for non-primitive types:
    if optional and not specify_optional(f"{path} {clas}"):
        return None

    # If it's in the registry, delegate to `configure_registrable`
    if clas in REGISTRY:
        return configure_registrable(full_path(clas), path)

    # Check for generic types
    try:
        origin = clas.__origin__
        args = clas.__args__
    except AttributeError:
        origin = args = None

    # List
    if origin in (List, Sequence):
        assert len(args) == 1
        list_type, = args
        print(f"list of {list_type}, enter blank to stop")
        values = []
        while True:
            value = configure(list_type, path=f"{path}.{len(values)}", optional=True)
            if value and value != DEFAULT:
                values.append(value)
            else:
                break
        return values
    # Dict
    elif origin == Dict:
        assert len(args) == 2
        key_type, value_type = args
        print(f"dict with keys of type {key_type} and values of type {value_type}")
        print("enter an empty key when done")
        result = {}
        while True:
            key = configure(key_type, path=f"{path}.{len(result)}.key", optional=True)
            if not key or key == DEFAULT:
                break
            value = configure(value_type, path=f"{path}.{len(result)}.value", optional=False)
            result[key] = value
        return result

    # Optional
    elif origin == Union and len(args) == 2 and type(None) in args:
        optional_type, _ = args
        assert optional_type != type(None)
        print(f"optional {optional_type}, leave empty for None")
        value = configure(optional_type, path=path, optional=True)
        if value and value != DEFAULT:
            return value
        else:
            return None
    # Union
    elif origin == Union:
        while True:
            print(f"union type, choose a number")
            for i, arg in enumerate(args):
                print(f"{i}: {arg}")
            choice = read(converter=int, optional=False)
            if 0 <= choice < len(args):
                union_type = args[choice]
                return configure(union_type, path=path, prompt=f"now I need a {union_type}")

    # Tuple
    elif origin == Tuple:
        values = []
        for i, arg in enumerate(args):
            print(f"tuple member {i} should be an {arg}")
            value = configure(arg, path=f"{path}.{i}")
        return values

    elif clas not in REGISTERED_CLASSES:
        print(f"unknown class {clas}, inserting TODO placeholder")
        return {"TODO": f"kwargs for {clas}"}

    config = {}
    registered_subclass = REGISTERED_CLASSES[clas]
    sig = registered_subclass.signature
    params = sig.parameters
    for param in params.values():
        param_name = param.name
        if param_name == "self":
            continue
        if param_name in skip_keys:
            continue

        if param_name in PROBABLY_SKIP:
            print(f"Although you can specify the {param_name} parameter, you probably want to skip it")
            skip = read("type FALSE if you don't want to skip", optional=True, converter=convert_bool)
            if skip != False:
                continue

        default = param.default
        annotation = param.annotation
        optional = (default != inspect._empty)

        default_str = ("(optional, default: " + str(default) + ")") if optional else ''
        prompt = f"{param_name} ({annotation}) {default_str}: "

        value = configure(annotation, f"{path}.{param_name}", prompt, optional)
        if not value or value == DEFAULT:
            # TODO
            continue
        config[param_name] = value
    return config

MODULE_NAMES = ['allennlp']#, 'torch']

if __name__ == "__main__":
    REGISTERED_CLASSES = {}
    for module_name in MODULE_NAMES:
        import_all_submodules(module_name)
        module = importlib.import_module(module_name)
        REGISTERED_CLASSES.update(collect_classes(module))
    for class_path, spec in PYTORCH_SIGS.items():
        REGISTERED_CLASSES[class_path] = RegisteredClass.from_spec(class_path, spec)

    REGISTRY = Registrable._registry

    model_config = configure_registrable('allennlp.models.model.Model', 'model')
    dataset_reader_config = configure_registrable('allennlp.data.dataset_readers.dataset_reader.DatasetReader', 'dataset_reader')
    iterator_config = configure_registrable('allennlp.data.iterators.data_iterator.DataIterator', 'iterator')
    train_data_path = configure(str, path='train_data_path')
    validation_data_path = configure(str, path='validation_data_path', optional=True)
    trainer_config = configure('allennlp.training.trainer.Trainer', 'trainer',
                               skip_keys=['model', 'optimizer', 'iterator', 'train_dataset', 'validation_dataset'])

    config = {
        'model': model_config,
        'trainer': trainer_config,
        'dataset_reader': dataset_reader_config,
        'iterator': iterator_config,
        'train_data_path': train_data_path
    }

    if validation_data_path:
        config['validation_data_path'] = validation_data_path

    print(config)

    print(json.dumps(config, indent=4))
