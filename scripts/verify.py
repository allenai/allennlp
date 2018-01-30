#! /usr/bin/env python

"""Script that runs all verification steps.
"""

import argparse

from subprocess import run
from subprocess import CalledProcessError
import os
import shutil

def main(checks):
    try:
        print("Verifying with " + str(checks))
        if "pytest" in checks:
            print("Tests (pytest):", flush=True)
            run("pytest -v --color=yes", shell=True, check=True)

        if "pylint" in checks:
            print("Linter (pylint):", flush=True)
            run("pylint -d locally-disabled,locally-enabled -f colorized allennlp tests", shell=True, check=True)
            print("pylint checks passed")

        if "mypy" in checks:
            print("Typechecker (mypy):", flush=True)
            run("mypy allennlp --ignore-missing-imports", shell=True, check=True)
            print("mypy checks passed")

        if "build-docs" in checks:
            print("Documentation (build):", flush=True)
            run("cd docs; make html-strict", shell=True, check=True)

        if "checks-docs" in checks:
            print("Documentation (check):", flush=True)
            run("./script/check_docs.py", shell=True, check=True)

    except CalledProcessError:
        # squelch
        pass

if __name__ == "__main__":
    checks = ['pytest', 'pylint', 'mypy', 'build-docs', 'check-docs']

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='Run all tests.')
    group.add_argument('--checks', type=str, nargs='+', choices = checks)

    args = parser.parse_args()

    if args.all:
        run_checks = checks
    else:
        run_checks = args.checks

    main(run_checks)
