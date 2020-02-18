#! /usr/bin/env python

"""Script that runs all verification steps.
"""

import argparse
from subprocess import run
from subprocess import CalledProcessError
import sys


def main(checks):
    try:
        print("Verifying with " + str(checks))
        if "pytest" in checks:
            print("Tests (pytest):", flush=True)
            run("pytest --color=yes -rf", shell=True, check=True)

        if "flake8" in checks:
            print("Linter (flake8)", flush=True)
            run("flake8 -v", shell=True, check=True)
            print("flake8 checks passed")

        if "black" in checks:
            print("Formatter (black)", flush=True)
            run("black -v --check .", shell=True, check=True)
            print("black checks passed")

        if "mypy" in checks:
            print("Typechecker (mypy):", flush=True)
            run(
                "mypy allennlp"
                # This is necessary because not all the imported libraries have type stubs.
                " --ignore-missing-imports"
                # This is necessary because PyTorch has some type stubs but they're incomplete,
                # and mypy will follow them and generate a lot of spurious errors.
                " --no-site-packages"
                # We are extremely lax about specifying Optional[] types, so we need this flag.
                # TODO: tighten up our type annotations and remove this
                " --no-strict-optional"
                # Some versions of mypy crash randomly when caching, probably because of our use of
                # NamedTuple (https://github.com/python/mypy/issues/7281).
                " --cache-dir=/dev/null",
                shell=True,
                check=True,
            )
            print("mypy checks passed")

        if "check-links" in checks:
            print("Checking links in Markdown files:", flush=True)
            run("./scripts/check_links.py", shell=True, check=True)
            print("check links passed")

        if "check-large-files" in checks:
            print("Checking all added files have size <= 2MB", flush=True)
            run("./scripts/check_large_files.sh 2", shell=True, check=True)
            print("check large files passed")

    except CalledProcessError:
        # squelch the exception stacktrace
        sys.exit(1)


if __name__ == "__main__":
    checks = [
        "pytest",
        "flake8",
        "mypy",
        "black",
        "check-links",
        "check-requirements",
        "check-large-files",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", default=checks, nargs="+", choices=checks)

    args = parser.parse_args()

    main(args.checks)
