FROM nvidia/cuda:10.0-base-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /stage/allennlp

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         python3.6-dev \
         python3-setuptools \
         python3-pip && \
     rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Copy select files needed for installing requirements.
# We only copy what we need here so small changes to the repository does not trigger re-installation of the requirements.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scripts/ scripts/
COPY allennlp/ allennlp/
COPY pytest.ini pytest.ini
COPY .flake8 .flake8
COPY tutorials/ tutorials/
COPY training_config training_config/
COPY setup.py setup.py
COPY README.md README.md

RUN pip install --editable .

# Compile EVALB - required for parsing evaluation.
# EVALB produces scary looking c-level output which we don't
# care about, so we redirect the output to /dev/null.
RUN cd allennlp/tools/EVALB && make &> /dev/null && cd ../../../

# Caching models when building the image makes a dockerized server start up faster, but is slow for
# running tests and things, so we skip it by default.
ARG CACHE_MODELS=false
RUN ./scripts/cache_models.py


# Optional argument to set an environment variable with the Git SHA
ARG SOURCE_COMMIT
ENV ALLENNLP_SOURCE_COMMIT $SOURCE_COMMIT

# Copy wrapper script to allow beaker to run resumable training workloads.
COPY scripts/ai2-internal/resumable_train.sh /stage/allennlp

LABEL maintainer="allennlp-contact@allenai.org"

EXPOSE 8000
CMD ["/bin/bash"]
