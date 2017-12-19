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

VERSION = '0.3.1-unreleased'


setup(name='allennlp',
      version=VERSION,
      description='An open-source NLP research library, built on PyTorch.',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='allennlp NLP deep learning machine reading',
      url='https://github.com/allenai/allennlp',
      author='Allen Institute for Artificial Intelligence',
      author_email='allennlp@allenai.org',
      license='Apache',
      packages=find_packages(),
      install_requires=[
          # Parameter parsing.
          'pyhocon==0.3.35',
          # Type checking for python
          'typing',
          # Adds an @overrides decorator for better documentation and error checking when using subclasses.
          'overrides',
          # Used by some old code.  We moved away from it because it's too slow, but some old code still
          # imports this.
          'nltk',
          # Mainly used for the faster tokenizer.
          'spacy>=2.0,<2.1',
          # Used by span prediction models.
          # Used in coreference resolution evaluation metrics.
          'scipy',
          'scikit-learn',
          # Training visualisation using tensorboard.
          'tensorboard',
          # Required by torch.utils.ffi
          'cffi==1.11.2',
          # Used by span prediction models.
          'numpy',
          # aws commandline tools for running on Docker remotely.
          'awscli>=1.11.91',
          # REST interface for models
          'sanic==0.6.0',
          'sanic-cors==0.6.0.2',
          # Talk to postgres demo database
          'psycopg2',
          # argument parsing for data cleaning scripts
          'argparse',
          # Used for downloading datasets over HTTP
          'requests>=2.18',
          # progress bars in data cleaning scripts
          'tqdm',
          # In SQuAD eval script, we use this to see if we likely have some tokenization problem.
          'editdistance',
          # Tutorial notebooks
          'jupyter',
          # For pretrained model weights
          'h5py',
          # For timezone utilities
          'pytz==2017.3'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      include_package_data=True,
      python_requires='>=3.6',
      zip_safe=False)
