from contextlib import contextmanager

from allennlp.common.util import ContextManagerFunctionReturnType, PathType, push_python_path, pushd

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
