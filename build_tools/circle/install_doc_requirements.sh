#!/usr/bin/env bash
set -x
set -e

# Nelson had this cleanup in here; not totally sure why.  We'll try it without.
# rm -rf ~/.pyenv && rm -rf ~/virtualenvs
# sudo -E apt-get -yq remove texlive-binaries --purge

# Installing required system packages to support the rendering of math
# notation in the HTML documentation
sudo -E apt-get -yq update
sudo -E apt-get -yq --no-install-suggests --no-install-recommends --force-yes \
     install dvipng texlive-latex-base texlive-latex-extra \
     texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Install dependencies with miniconda
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
if [[ ! -f miniconda.sh ]]
then
   wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
   -O miniconda.sh
fi
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
cd ..
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda
popd

# Configure the conda environment and put it in the path using the
# provided versions.
conda create -n $CONDA_ENV_NAME --yes --quiet python=3.5.2
source activate $CONDA_ENV_NAME

# Install pip dependencies.
pip install -r requirements.txt
pip install -r requirements_test.txt
