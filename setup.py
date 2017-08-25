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
import os
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

VERSION = '0.1a6'


def gen_data_files(*directories):
    results = []
    for src_dir in directories:
        for root, dirs, files in os.walk(src_dir):
            results.append((root, map(lambda f: root + "/" + f, files)))
    return results

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
          'pyhocon==0.3.35',
          'typing',
          'overrides',
          'nltk',
          'spacy',
          'numpy',
          'pillow',
          'tensorboard-pytorch',
          'awscli>=1.11.91',
          'sanic==0.6.0',
          'argparse',
          'requests>=2.18',
          'tqdm',
          'jupyter'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      include_package_data=True,
      python_requires='~=3.5',
      zip_safe=False)
