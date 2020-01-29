from contextlib import contextmanager

from pip._internal.cli.main import main as pip_main

from allennlp.common.util import ContextManagerFunctionReturnType, PathType, push_python_path, pushd


@contextmanager
def pip_install(path: PathType, package_name: str) -> ContextManagerFunctionReturnType[None]:
    """
    Installs a package with pip located in the given path and with the given name.

    This method is intended to use with `with`, so after its usage, the package will be
    uninstalled.
    """
    pip_main(["install", str(path)])
    try:
        yield
    finally:
        pip_main(["uninstall", "-y", package_name])


@contextmanager
def push_python_project(path: PathType) -> ContextManagerFunctionReturnType[None]:
    """
    Changes the current directory to the given path and prepends it to `sys.path`.

    It simulates the behavior of running a command from a Python's project root directory,
    which is part of Python's path.

    This method is intended to use with `with`, so after its usage, the current directory will be
    set to the previous value and its value removed from `sys.path`.
    """
    with pushd(path), push_python_path("."):
        yield
