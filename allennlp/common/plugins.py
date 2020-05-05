"""
Plugin management.

AllenNLP supports loading "plugins" dynamically. A plugin is just a Python package that
can be found and imported by AllenNLP. This is done by creating a file named `.allennlp_plugins`
in the directory where the `allennlp` command is run that lists the modules that should be loaded,
one per line.
"""

import importlib
import logging
import os
from typing import Iterable

from allennlp.common.util import push_python_path, import_module_and_submodules

logger = logging.getLogger(__name__)


DEFAULT_PLUGINS = ("allennlp_models",)


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
        yield from discover_file_plugins()


def import_plugins() -> None:
    """
    Imports the plugins found with `discover_plugins()`.
    """
    for module in DEFAULT_PLUGINS:
        try:
            # For default plugins we recursively import everything.
            import_module_and_submodules(module)
        except ModuleNotFoundError:
            pass
    for module_name in discover_plugins():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            logger.error(f"Plugin {module_name} could not be loaded: {e}")
