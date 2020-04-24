# This Dockerfile creates an environment suitable for downstream usage of AllenNLP.
# It's built from a wheel installation of allennlp.
#
# To build this image you will need a wheel of allennlp in the `dist/` directory.
# You can either obtain a pre-built wheel from a PyPI release or build a new wheel from
# source.
#
# PyPI release wheels can be downloaded by going to https://pypi.org/project/allennlp/#history,
# clicking on the desired release, and then clicking "Download files" in the left sidebar.
# After downloading, make you sure you put the wheel in the `dist/` directory
# (which may not exist if you haven't built a wheel from source yet).
#
# To build a wheel from source, just run `python setup.py wheel`.
#
# *Before building the image, make sure you only have one wheel in the `dist/` directory.*
#
# Once you have your wheel, run `make docker-image`. By default this builds an image
# with the tag `allennlp/allennlp`. You can change this to anything you want
# by setting the `DOCKER_TAG` flag when you call `make`. For example,
# `make docker-image DOCKER_TAG=my-allennlp`.

FROM python:3.6.10-stretch

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage/allennlp

# Install the wheel of AllenNLP.

COPY dist dist/
RUN pip install $(ls dist/*.whl)

# Copy wrapper script to allow beaker to run resumable training workloads.
COPY scripts/ai2_internal/resumable_train.sh /stage/allennlp

LABEL maintainer="allennlp-contact@allenai.org"

ENTRYPOINT ["allennlp"]
