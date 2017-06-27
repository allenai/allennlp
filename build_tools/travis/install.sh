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
    # If we are running pylint, use Python 3.5.2 due to
    # bug in pylint. https://github.com/PyCQA/pylint/issues/1295
    conda create -n testenv352 --yes python=3.5.2
    conda create -n testenv --yes python=3.5
fi
cd ..
popd

# Activate the python environment we created.
if [[ "$RUN_PYLINT" == "true" ]]; then
    source activate testenv352
else
    source activate testenv
fi

# Install requirements via pip and download data inside our conda environment.
bash scripts/install_requirements.sh

# List the packages to get their versions for debugging
pip list
