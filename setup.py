"""
In order to create a package for pypi, you need to follow several steps.

1. Create a .pypirc in your home directory. It should look like this:

```
[distutils]
index-servers =
  pypi
  pypitest

[pypi]
username=allennlp
password= Get the password from LastPass.

[pypitest]
repository=https://test.pypi.org/legacy/
username=allennlp
password= Get the password from LastPass.
```
run chmod 600 ./pypirc so only you can read/write.

1. Change the version in docs/conf.py and setup.py.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level allennlp directory.
   (this will build a wheel for the python version you use to build it - make sure you use python 3.x).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions of allennlp.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi allennlp

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

"""
from setuptools import setup, find_packages
import sys

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
VERSION = {}
with open("allennlp/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# make pytest-runner a conditional requirement,
# per: https://github.com/pytest-dev/pytest-runner#considerations
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup_requirements = [
    # add other setup requirements as necessary
] + pytest_runner

setup(name='allennlp',
      version=VERSION["VERSION"],
      description='An open-source NLP research library, built on PyTorch.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='allennlp NLP deep learning machine reading',
      url='https://github.com/allenai/allennlp',
      author='Allen Institute for Artificial Intelligence',
      author_email='allennlp@allenai.org',
      license='Apache',
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      install_requires=[
          'torch>=0.4.1',
          "jsonnet>=0.10.0 ; sys.platform != 'win32'",
          'overrides',
          'nltk',
          'spacy>=2.0.18,<2.2',
          'numpy',
          'tensorboardX>=1.2',
          'awscli>=1.11.91',
          'boto3',
          'moto>=1.3.4',
          'flask>=1.0.2',
          'flask-cors>=3.0.7',
          'gevent>=1.3.6',
          'requests>=2.18',
          'tqdm>=4.19',
          'editdistance',
          'h5py',
          'scikit-learn',
          'scipy',
          'pytz>=2017.3',
          'unidecode',
          'matplotlib>=2.2.3',
          'pytest',
          'flaky',
          'responses>=0.7',
          'numpydoc>=0.8.0',
          'conllu==0.11',
          'parsimonious>=0.8.0',
          'ftfy',
          'sqlparse>=0.2.4',
          'word2number>=1.1',
          'pytorch-pretrained-bert>=0.6.0',
      ],
      entry_points={
          'console_scripts': [
              "allennlp=allennlp.run:run"
          ]
      },
      setup_requires=setup_requirements,
      tests_require=[
          'pytest',
          'flaky',
          'responses>=0.7',
          'moto==1.3.4',
      ],
      include_package_data=True,
      python_requires='>=3.6.1',
      zip_safe=False)
