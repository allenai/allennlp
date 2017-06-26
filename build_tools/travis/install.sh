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

# Install requirements via pip in our conda environment
pip install -U -r requirements.txt
pip install -q http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl

# List the packages to get their versions for debugging
pip list

# Install punkt tokenizer
python -m nltk.downloader punkt

# Install spacy data
python -m spacy.en.download all
