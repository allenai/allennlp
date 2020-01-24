import importlib
import os
import pkgutil
from typing import Iterable


def discover_namespace_plugins() -> Iterable[pkgutil.ModuleInfo]:
    try:
        import allennlp_plugins as namespace

        return pkgutil.iter_modules(namespace.__path__, namespace.__name__ + ".")
    except ImportError:
        return []


def discover_file_plugins() -> Iterable[str]:
    plugins_filename = ".allennlp_plugins"
    if os.path.isfile(plugins_filename):
        with open(plugins_filename) as file_:
            for module_name in file_.readlines():
                module_name = module_name.strip()
                if module_name:
                    yield module_name
    else:
        return []


def discover_plugins() -> Iterable[str]:
    for module_info in discover_namespace_plugins():
        yield module_info.name
    yield from discover_file_plugins()


def import_plugins() -> None:
    for module_name in discover_plugins():
        importlib.import_module(module_name)
