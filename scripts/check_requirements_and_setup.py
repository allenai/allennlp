#!/usr/bin/env python

"""
Checks that the requirements as specifed in requirements.txt and in setup.py are identical.
"""

import re
import sys
from typing import Set, Dict, Tuple, Optional


PackagesType = Dict[str, Optional[str]]  # pylint: disable=invalid-name


def parse_section_name(line: str) -> str:
    return line.replace("####", "").strip()


def parse_package(line: str) -> Tuple[str, Optional[str]]:
    parts = re.split(r"(==|>=|<=|>|<)", line)
    module = parts[0]
    version = line.replace(module, "")
    return (module, version)


def parse_requirements() -> Tuple[PackagesType, PackagesType, Set[str]]:
    """Parse all dependencies out of the requirements.txt file."""
    essential_packages: PackagesType = {}
    other_packages: PackagesType = {}
    duplicates: Set[str] = set()
    with open("requirements.txt", "r") as req_file:
        section: str = ""
        for line in req_file:
            line = line.strip()

            if line.startswith("####"):
                # Line is a section name.
                section = parse_section_name(line)
                continue

            if not line or line.startswith("#"):
                # Line is empty or just regular comment.
                continue

            module, version = parse_package(line)
            if module in essential_packages or module in other_packages:
                duplicates.add(module)

            if section.startswith("ESSENTIAL"):
                essential_packages[module] = version
            else:
                other_packages[module] = version

    return essential_packages, other_packages, duplicates


def parse_setup() -> Tuple[PackagesType, PackagesType, Set[str]]:
    """Parse all dependencies out of the setup.py script."""
    essential_packages: PackagesType = {}
    test_packages: PackagesType = {}
    duplicates: Set[str] = set()
    with open('setup.py') as setup_file:
        contents = setup_file.read()

    # Parse out essential packages.
    package_string = re.search(r"""install_requires=\[[\s\n]*['"](.*?)['"][\s\n]*\]""",
                               contents, re.DOTALL).groups()[0].strip()
    for package in re.split(r"""['"],[\s\n]+['"]""", package_string):
        module, version = parse_package(package)
        if module in essential_packages:
            duplicates.add(module)
        else:
            essential_packages[module] = version

    # Parse packages only needed for testing.
    package_string = re.search(r"""tests_require=\[[\s\n]*['"](.*?)['"][\s\n]*\]""",
                               contents, re.DOTALL).groups()[0].strip()
    for package in re.split(r"""['"],[\s\n]+['"]""", package_string):
        module, version = parse_package(package)
        if module in essential_packages or module in test_packages:
            duplicates.add(module)
        else:
            test_packages[module] = version

    return essential_packages, test_packages, duplicates


def main() -> int:
    exit_code = 0

    requirements_essential, requirements_other, duplicates = parse_requirements()
    if duplicates:
        exit_code = 1
        for module in duplicates:
            print(f"  ✗ '{module}' appears more than once in requirements.txt")

    setup_essential, setup_test, duplicates = parse_setup()
    if duplicates:
        exit_code = 1
        for module in duplicates:
            print(f"  ✗ '{module}' appears more than once in setup.py")

    # Find all packages listed as essential in requirements.txt that differ
    # in or are absent from setup.py.
    for module, version in requirements_essential.items():
        if module not in setup_essential and module not in setup_test:
            exit_code = 1
            print(f"  ✗ '{module}' listed as essential in requirements.txt "
                  f"but is missing from setup.py")
        elif module in setup_test:
            exit_code = 1
            print(f"  ✗ '{module}' listed as essential in requirements.txt "
                  f"but is only listed a test requirement in setup.py")
        elif setup_essential[module] != version:
            exit_code = 1
            print(f"  ✗ '{module}' has version '{version}' in requirements.txt "
                  f"but '{setup_essential[module]}' in setup.py")

    # Find all packages listed as essential setup.py that are not listed as essential
    # in requirements.txt.
    for module, version in setup_essential.items():
        if module in requirements_other:
            exit_code = 1
            print(f"  ✗ '{module}' appears in setup.py but is listed as "
                  f"non-essential in requirements.txt")
        elif module not in requirements_essential:
            exit_code = 1
            print(f"  ✗ '{module}' appears in setup.py but not in requirements.txt")

    # Find all packages listed under `tests_require` in setup.py that are missing
    # from requirements.txt.
    for module, version in setup_test.items():
        if module not in requirements_other:
            exit_code = 1
            print(f"  ✗ '{module}' appears in as a test requirement in setup.py "
                  f"but is missing from requirements.txt")
        elif requirements_other[module] != version:
            exit_code = 1
            print(f"  ✗ '{module}' has version '{version}' in setup.py "
                  f"but '{requirements_other[module]}' in requirements.txt")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
