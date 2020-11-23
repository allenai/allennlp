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

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = {}  # type: ignore
with open("allennlp/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="allennlp",
    version=VERSION["VERSION"],
    description="An open-source NLP research library, built on PyTorch.",
    long_description=open("README.md").read(),
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
    install_requires=[
        "torch>=1.6.0,<1.8.0",
        "jsonnet>=0.10.0 ; sys.platform != 'win32'",
        "overrides==3.1.0",
        "nltk",
        "spacy>=2.1.0,<2.4",
        "numpy",
        "tensorboardX>=1.2",
        "boto3>=1.14,<2.0",
        "requests>=2.18",
        "tqdm>=4.19",
        "h5py",
        "scikit-learn",
        "scipy",
        "pytest",
        "transformers>=3.4,<3.6",
        "jsonpickle",
        "dataclasses;python_version<'3.7'",
        "filelock>=3.0,<3.1",
    ],
    entry_points={"console_scripts": ["allennlp=allennlp.__main__:run"]},
    include_package_data=True,
    python_requires=">=3.6.1",
    zip_safe=False,
)
