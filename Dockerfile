FROM python:3.6.3-jessie

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /usr/local/nvidia/bin/:$PATH
ENV PYTHONHASHSEED 2157

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

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
COPY training_config training_config/

# Run tests to verify the Docker build
RUN PYTHONDONTWRITEBYTECODE=1 pytest

# Add model caching
ARG CACHE_MODELS=false
RUN ./scripts/cache_models.py

LABEL maintainer="allennlp-contact@allenai.org"

EXPOSE 8000
CMD ["/bin/bash"]
