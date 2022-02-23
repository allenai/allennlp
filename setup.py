from collections import defaultdict
from setuptools import find_packages, setup

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release


def parse_requirements_file(path, allowed_extras: set = None, include_all_extra: bool = True):
    requirements = []
    extras = defaultdict(list)
    with open(path) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            req, *needed_by = line.split("# needed by:")
            req = fix_url_dependencies(req.strip())
            if needed_by:
                for extra in needed_by[0].strip().split(","):
                    extra = extra.strip()
                    if allowed_extras is not None and extra not in allowed_extras:
                        raise ValueError(f"invalid extra '{extra}' in {path}")
                    extras[extra].append(req)
                if include_all_extra and req not in extras["all"]:
                    extras["all"].append(req)
            else:
                requirements.append(req)
    return requirements, extras


integrations = {"checklist"}

# Load requirements.
install_requirements, extras = parse_requirements_file(
    "requirements.txt", allowed_extras=integrations
)
dev_requirements, dev_extras = parse_requirements_file(
    "dev-requirements.txt", allowed_extras={"examples"}, include_all_extra=False
)
extras["dev"] = dev_requirements
extras.update(dev_extras)

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = {}  # type: ignore
with open("allennlp/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="allennlp",
    version=VERSION["VERSION"],
    description="An open-source NLP research library, built on PyTorch.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="allennlp NLP deep learning machine reading",
    url="https://github.com/allenai/allennlp",
    author="Allen Institute for Artificial Intelligence",
    author_email="allennlp@allenai.org",
    license="Apache",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "test_fixtures",
            "test_fixtures.*",
            "benchmarks",
            "benchmarks.*",
        ]
    ),
    install_requires=install_requirements,
    extras_require=extras,
    entry_points={"console_scripts": ["allennlp=allennlp.__main__:run"]},
    include_package_data=True,
    python_requires=">=3.7.1",
    zip_safe=False,
)
