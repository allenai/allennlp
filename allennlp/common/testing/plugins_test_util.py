from contextlib import contextmanager

from pip._internal.cli.main import main as pip_main

from allennlp.common.util import ContextManagerFunctionReturnType, PathType, push_python_path, pushd


@contextmanager
def pip_install(path: PathType, package_name: str) -> ContextManagerFunctionReturnType[None]:
    pip_main(["install", str(path)])
    try:
        yield
    finally:
        pip_main(["uninstall", "-y", package_name])


@contextmanager
def push_python_project(path: PathType) -> ContextManagerFunctionReturnType[None]:
    # In general when we run scripts or commands in a project, the current directory is the root of it
    # and is part of the path. So we emulate this here with `push_python_path`.
    with pushd(path), push_python_path("."):
        yield
