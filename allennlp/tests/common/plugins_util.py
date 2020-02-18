import subprocess
import sys
from contextlib import contextmanager

from allennlp.common.util import ContextManagerFunctionReturnType, PathType, push_python_path, pushd


@contextmanager
def pip_install(path: PathType, package_name: str) -> ContextManagerFunctionReturnType[None]:
    """
    Installs a package with pip located in the given path and with the given name.

    This method is intended to use with `with`, so after its usage, the package will be
    uninstalled.
    """
    # Not using the function `main` from pip._internal because it assumes that once it finished,
    # the process will terminate, and thus it can failed if called multiple times. See
    # https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
    # It actually fails in pip==19.3.1 if called multiple times in the same process (but it works
    # in 20.0).
    # Starting a new process is slower, but it's not a problem if it's not called often.
    subprocess.check_call([sys.executable, "-m", "pip", "install", str(path)])
    try:
        yield
    finally:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])


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
