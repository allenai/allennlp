#!/bin/bash
set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Add the miniconda bin directory to $PATH
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f /home/travis/miniconda3/bin/activate ]]
    then
    if [[ ! -f miniconda.sh ]]
        then
            wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
                -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda update --yes conda
    conda create -n testenv --yes python=3.6
fi
cd ..
popd

# Activate the python environment we created.
source activate testenv

# Install requirements via pip and download data inside our conda environment.
INSTALL_TEST_REQUIREMENTS="true" bash scripts/install_requirements.sh
pip install --no-cache-dir -q http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl

# List the packages to get their versions for debugging
pip list
