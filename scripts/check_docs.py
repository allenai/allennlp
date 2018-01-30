#! /usr/bin/env python
# pylint: disable=invalid-name,redefined-outer-name
from glob import glob
import os
import re
import sys
from typing import Set

DOCS_DIR = 'doc/api'
MODULE_REGEX = r"\ballennlp\.[a-z0-9_.]+\b"
MODULE_GLOB = 'allennlp/**/*.py'

MODULES_THAT_NEED_NO_DOCS: Set[str] = {
        # No docs at top level.
        'allennlp',
        'allennlp.version',

        # No docs for custom extensions, which aren't even in python.
        'allennlp.custom_extensions',
        'allennlp.custom_extensions._ext',
        'allennlp.custom_extensions._ext.highway_lstm_layer',
        'allennlp.custom_extensions.build',

        # TODO(joelgrus): Figure out how to make these docs build;
        # the cffi part is causing problems.
        'allennlp.modules.alternating_highway_lstm',
        # Private base class, no docs needed.
        'allennlp.modules.encoder_base',
        # TODO(Mark): remove this once the coref model is switched over.
        'allennlp.models.coreference_resolution.coref_v2'
}

DOCS_THAT_NEED_NO_MODULES: Set[str] = {
        # Function is defined in allennlp/commands/__init__.py.
        'allennlp.commands.main',
}


def documented_modules(docs_dir: str = DOCS_DIR, module_regex: str = MODULE_REGEX) -> Set[str]:
    modules: Set[str] = set()

    for path in glob(os.path.join(docs_dir, '**/*.rst'), recursive=True):
        with open(path) as f:
            text = f.read()
            for module in re.findall(module_regex, text):
                modules.add(module)

    return modules


def existing_modules(module_glob: str = MODULE_GLOB) -> Set[str]:
    modules: Set[str] = set()

    for path in glob(module_glob, recursive=True):
        path = re.sub(".py$", "", path)
        path = re.sub("/__init__", "", path)
        path = path.replace("/", ".")
        modules.add(path)

    return modules

if __name__ == "__main__":
    success = True
    existing = existing_modules()
    documented = documented_modules()
    for module in sorted(existing):
        if module not in documented and module not in MODULES_THAT_NEED_NO_DOCS:
            print("undocumented module:", module)
            success = False
    for module in sorted(documented):
        if module not in existing and module not in DOCS_THAT_NEED_NO_MODULES:
            print("documented but nonexistent:", module)
            success = False

    if not success:
        sys.exit(1)
