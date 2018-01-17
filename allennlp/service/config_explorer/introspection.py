import inspect
import importlib
import pkgutil
from types import ModuleType
from typing import NamedTuple, Dict, Sequence, Callable, Optional, TypeVar, List, Union, Sequence, Tuple

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

def import_all_submodules(module_name: str):
    """
    Import all submodules of the specified module
    """
    module = importlib.import_module(module_name)

    for _loader, name, _is_pkg in pkgutil.walk_packages(module.__path__):
        importlib.import_module(f"{module_name}.{name}")


def collect_classes(module: ModuleType,
                    result: Dict[str, RegisteredClass] = None) -> Dict[str, RegisteredClass]:
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
                    key not in result):
            under_registrable = obj.__base__.__name__ == "Registrable"
            signature = inspect.signature(obj.__init__)
            print(obj, signature)
            reg = RegisteredClass(obj.__module__, obj.__name__, under_registrable, signature, obj)
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

def _read_bool(s: str) -> bool:
    """
    using bool(s) gives the wrong result a lot of the time
    """
    if s.lower() in ('true', 't'):
        return True
    elif s.lower() in ('false', 'f'):
        return False
    else:
        raise ValueError("please input true or false")

def configure_registrable(class_name: str) -> JsonDict:
    # Get the list of valid types
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

    config = {'type': typ}

    subclass = REGISTRY[clas][typ]
    subclass_name = full_path(subclass)
    registered_subclass = REGISTERED_CLASSES[subclass_name]
    sig = registered_subclass.signature
    params = sig.parameters
    for param in params.values():
        param_name = param.name
        if param_name == "self":
            continue
        default = param.default
        annotation = param.annotation
        optional = (default != inspect._empty)

        default_str = ("(default: " + str(default) + ")") if optional else ''
        prompt = f"{param_name} ({annotation}) {default_str}: "

        if annotation in REGISTRY:
            # This is another registrable class, so make a recursive call to configure it
            value = configure_registrable(full_path(annotation))
        elif annotation in (int, str, float):
            # Built-in type, so just deal with it
            value = read(prompt, optional=optional, converter=annotation)
        elif annotation == bool:
            # bool needs special handling, because bool("False") == True [!]
            value = read(prompt, optional=optional, converter=_read_bool)
        elif annotation.__origin__ == Dict:
            # This is a dict, so we need to accept key-value pairs.
            print(prompt)

            read_dict = True
            if optional:
                # Check if we want to use the default value
                use_default = read("use default value? (type 'no' if not)", optional=True)
                if use_default == DEFAULT or use_default.lower() in ('y', "yes"):
                    # Just use the default value
                    value = DEFAULT
                    read_dict = False

            if read_dict:
                dict_value = {}
                key_type, value_type = annotation.__args__
                while True:
                    print("next entry (leave blank to finish)")
                    key = read(f"key: ", optional=True, converter=key_type)
                    if key == DEFAULT:
                        break
                    elif value_type in (int, float, str):
                        value = read(f"value:", optional=False, converter=value_type)
                    elif value_type == bool:
                        value = read(f"value:", optional=False, converter=_read_bool)
                    elif isinstance(value_type, Registrable):
                        value = configure_registrable(full_path(value_type))
                    else:
                        raise RuntimeError(f"unconfigurable type: {value_type}")
                    dict_value[key] = value
                value = dict_value

        elif annotation.__origin__ == List:
            print(prompt)

            read_list = True
            if optional:
                use_default = read("use default value? (type 'no' if not)")
                if use_default is None or use_default.lower() in ('y', "yes"):
                    value = DEFAULT
                    read_list = False

            if read_list:
                list_value = []
                value_type, = annotation.__args__
                while True:
                    print("next entry (leave blank to finish)")
                    if value_type in (int, float, str):
                        value = read(f"value:", optional=True, converter=value_type)
                    elif value_type == bool:
                        value = read(f"value:", optional=True, converter=_read_bool)
                    else:
                        raise RuntimeError(f"list of {value_type}?")
                    if value == DEFAULT:
                        break
                    else:
                        list_value.append(value)
                value = list_value

        else:
            raise RuntimeError(f"unconfigurable type: {annotation}")

        if value != DEFAULT:
            config[param_name] = value

    return config

def get_info(registered_classes: dict,
             annotation: type):
    origin = getattr(annotation, '__origin__', None)

    if hasattr(annotation, '__name__'):
        name = full_path(annotation)
    else:
        name = None

    if name in registered_classes:
        return name
    elif annotation in (int, bool, float, str):
        return annotation.__name__
    elif annotation == type(None):
        return "None"
    elif origin == Dict:
        key_type, value_type = annotation.__args__
        return ['dict',
                get_info(registered_classes, key_type),
                get_info(registered_classes, value_type)]
    elif origin in (List, Sequence):
        value_type, = annotation.__args__
        return ['list',
                get_info(registered_classes, value_type)]
    elif origin == Union:
        return ['union'] + [get_info(registered_classes, arg) for arg in annotation.__args__]
    elif origin == Tuple:
        return ['tuple'] + [get_info(registered_classes, arg) for arg in annotation.__args__]
    else:
        return "unknown"

def get_infos(registered_classes: dict) -> JsonDict:
    infos = {}

    for class_name, registered_subclass in registered_classes.items():
        signature = registered_subclass.signature
        params = signature.parameters

        config_infos: List[JsonDict] = []

        for param in params.values():
            param_name = param.name
            annotation = param.annotation

            if param_name == 'self' or annotation == inspect._empty:
                continue

            default = param.default
            optional = (default != inspect._empty)

            print(class_name, param_name, default, annotation)

            info: JsonDict = {
                'name': param_name,
                'default': str(default) if optional else None,
                'optional': optional,
                'annotation': get_info(registered_classes, annotation)
            }

            config_infos.append(info)

        infos[class_name] = config_infos

    return infos


MODULE_NAMES = ['allennlp']

if __name__ == "__main__":
    REGISTERED_CLASSES = {}
    for module_name in MODULE_NAMES:
        import_all_submodules(module_name)
        module = importlib.import_module(module_name)
        REGISTERED_CLASSES.update(collect_classes(module))
    REGISTRY = Registrable._registry
