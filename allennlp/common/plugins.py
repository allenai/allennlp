import importlib
import os
import pkgutil
import sys
from typing import Iterable


def discover_namespace_plugins(
    namespace_name: str = "allennlp_plugins",
) -> Iterable[pkgutil.ModuleInfo]:
    try:
        reload = namespace_name in sys.modules

        namespace_module = importlib.import_module(namespace_name)

        if reload:
            importlib.reload(namespace_module)

        return pkgutil.iter_modules(
            namespace_module.__path__, namespace_module.__name__ + "."
        )  # type: ignore
    except ModuleNotFoundError:
        return []


def discover_file_plugins(plugins_filename: str = ".allennlp_plugins") -> Iterable[str]:
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
