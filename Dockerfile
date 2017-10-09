FROM ubuntu:16.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# Add nvidia and conda binaries to path.
ENV PATH /usr/local/nvidia/bin/:/opt/conda/bin:$PATH
ENV PYTHONHASHSEED 2157

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:$LD_LIBRARY_PATH

WORKDIR /stage

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
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
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Anaconda.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Use python 3.6
RUN conda install python=3.6

# Install npm
RUN curl -sL https://deb.nodesource.com/setup_8.x | bash - && apt-get install -y nodejs

# Copy select files needed for installing requirements.
# We only copy what we need here so small changes to the repository does not trigger re-installation of the requirements.
COPY requirements.txt .
COPY requirements_test.txt .
COPY scripts/install_requirements.sh scripts/install_requirements.sh
RUN INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh
RUN pip install --no-cache-dir -q http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl

# Build demo
COPY demo/ demo/
RUN cd demo && npm install && npm run build && cd ..

COPY allennlp/ allennlp/
COPY tests/ tests/
COPY pytest.ini pytest.ini
COPY scripts/ scripts/
COPY tutorials/ tutorials/
COPY experiment_config experiment_config/

# Run tests to verify the Docker build
RUN PYTHONDONTWRITEBYTECODE=1 pytest

LABEL maintainer="allennlp-contact@allenai.org"

EXPOSE 8000
CMD ["/bin/bash"]
