# This Dockerfile creates an environment suitable for downstream usage of AllenNLP.
# It's built from a wheel installation of allennlp.

FROM python:3.7

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

# Install the wheel of AllenNLP.
COPY dist dist/
RUN pip install $(ls dist/*.whl)

# Copy wrapper script to allow beaker to run resumable training workloads.
COPY scripts/ai2_internal/resumable_train.sh /stage/allennlp

LABEL maintainer="allennlp-contact@allenai.org"

ENTRYPOINT ["allennlp"]
