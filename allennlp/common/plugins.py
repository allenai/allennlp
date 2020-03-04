"""
Plugin management.

AllenNLP supports loading "plugins" dynamically. A plugin is just a Python package that is found
by AllenNLP with the methods provided in this module.

There are two ways of declaring plugins for discovery:

    * Writing the package name to import in a file (typically ".allennlp_plugins", in the current
    directory from which the command "allennlp" is run). This is the simplest approach.

    * Creating a folder called "allennlp_plugins" that's in the Python path when you run the
    "allennlp" command (typically under your project's root directory), then creating a subfolder
    with the name you want and creating an "__init__.py" file that imports the code you want (e.g.,
    your Python package). This option is preferred when you want to create a pip-installable
    package and you want to make your AllenNLP plugin available when users install your package.
    See [allennlp-server](https://github.com/allenai/allennlp-server) for an example.
"""
import importlib
import logging
import os
import pkgutil
import sys
from typing import Iterable

from allennlp.common.util import push_python_path

logger = logging.getLogger(__name__)


def discover_namespace_plugins(
    namespace_name: str = "allennlp_plugins",
) -> Iterable[pkgutil.ModuleInfo]:
    """
    Returns an iterable of the plugins found, declared within the namespace package `namespace_name`.
    """
    try:
        reload = namespace_name in sys.modules

        namespace_module = importlib.import_module(namespace_name)

        if reload:
            importlib.reload(namespace_module)

        return pkgutil.iter_modules(
            namespace_module.__path__, namespace_module.__name__ + "."  # type: ignore
        )
    except ModuleNotFoundError:
        return []


def discover_file_plugins(plugins_filename: str = ".allennlp_plugins") -> Iterable[str]:
    """
    Returns an iterable of the plugins found, declared within a file whose path is `plugins_filename`.
    """
    if os.path.isfile(plugins_filename):
        with open(plugins_filename) as file_:
            for module_name in file_.readlines():
                module_name = module_name.strip()
                if module_name:
                    yield module_name
    else:
        return []


def discover_plugins() -> Iterable[str]:
    """
    Returns an iterable of the plugins found.
    """
    with push_python_path("."):
        for module_info in discover_namespace_plugins():
            yield module_info.name
        yield from discover_file_plugins()


def import_plugins() -> None:
    """
    Imports the plugins found with `discover_plugins()`.
    """
    for module_name in discover_plugins():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            logger.error(f"Plugin {module_name} could not be loaded: {e}")
