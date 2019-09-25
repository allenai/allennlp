#!/bin/bash

# Sets up an Ubuntu box. Tested against Google Cloud image xxx.
# Derived from https://github.com/allenai/allennlp/blob/master/Dockerfile

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Install base packages (excluding git)
apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential \
    openjdk-8-jdk

curl https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh > ~/Anaconda3-5.2.0-Linux-x86_64.sh
chmod +x ~/Anaconda3-5.2.0-Linux-x86_64.sh
~/Anaconda3-5.2.0-Linux-x86_64.sh

conda create -n allennlp python=3.6
source activate allennlp

git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
mkdir -p ~/repos/brendanr
cd ~/repos/brendanr
git clone git@github.com:brendan-ai2/allennlp.git
cd allennlp

# Compile EVALB - required for parsing evaluation.
# EVALB produces scary looking c-level output which we don't
# care about, so we redirect the output to /dev/null.
cd allennlp/tools/EVALB && make &> /dev/null && cd ../../../
