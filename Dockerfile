# This Dockerfile creates an environment suitable for downstream usage of AllenNLP.
# It's built from a wheel installation of allennlp.

FROM python:3.8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage/allennlp

# Install torch ecosystem first. This build arg should be in the form of a version requirement,
# like 'torch==1.7' or 'torch==1.7+cu102 -f https://download.pytorch.org/whl/torch_stable.html'.
ARG TORCH
RUN pip install --no-cache-dir ${TORCH}

# Installing AllenNLP's dependencies is the most time-consuming part of building
# this Docker image, so we make use of layer caching here by adding the minimal files
# necessary to install the dependencies.
COPY allennlp/version.py allennlp/version.py
COPY setup.py .
RUN touch allennlp/__init__.py \
    && touch README.md \
    && pip install --no-cache-dir -e .

# Now add the full package source and re-install just the package.
COPY allennlp allennlp
RUN pip install --no-cache-dir --no-deps -e .

WORKDIR /app/

# Copy wrapper script to allow beaker to run resumable training workloads.
COPY scripts/ai2_internal/resumable_train.sh .

LABEL maintainer="allennlp-contact@allenai.org"

ENTRYPOINT ["allennlp"]
