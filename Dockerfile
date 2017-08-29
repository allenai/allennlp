ARG from=nvidia/cuda:8.0-cudnn5-devel
FROM $from

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV PYTHONHASHSEED 2157

# Override Nvidia's default LD paths, since they're misconfigured in the base image.
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

WORKDIR /stage
EXPOSE 8000
CMD ["/bin/bash"]

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
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

# Use python 3.5
RUN conda install python=3.5

# Copy select files needed for installing requirements.
# We only copy what we need here so small changes to the repository does not trigger re-installation of the requirements.
COPY requirements.txt .
COPY requirements_test.txt .
COPY scripts/install_requirements.sh scripts/install_requirements.sh
RUN INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh
RUN pip install --no-cache-dir -q http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl

COPY allennlp/ allennlp/
COPY tests/ tests/
COPY pytest.ini pytest.ini
COPY scripts/ scripts/
COPY tutorials/ tutorials/

# Run tests to verify the Docker build
RUN PYTHONDONTWRITEBYTECODE=1 pytest
