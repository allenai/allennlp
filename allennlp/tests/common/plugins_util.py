import subprocess
import sys
from contextlib import contextmanager

from allennlp.common.util import ContextManagerFunctionReturnType, PathType


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
