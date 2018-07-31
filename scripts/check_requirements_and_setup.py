#! /usr/bin/env python
"""
Checks that the requirements as specifed in requirements.txt
and in setup.py are identical.
"""

import re
import sys

with open('requirements.txt') as f:
    requirements = {line.strip() for line in f if line.strip() and not line.startswith('#')}

with open('setup.py') as f:
    install_requires_regex = r"install_requires=\[(.*?)\]"
    setup_requirements = re.search(install_requires_regex, f.read(), re.DOTALL).groups()[0]

    requirement_regex = r"'(.+?)'"
    setup_requirements = set(re.findall(requirement_regex, setup_requirements))

if not requirements:
    print("no requirements found!")
    sys.exit(1)

if requirements != setup_requirements:
    print("requirements mismatch")
    print(f"in requirements.txt: {requirements - setup_requirements}")
    print(f"in setup.py: {setup_requirements - requirements}")
    sys.exit(-1)
