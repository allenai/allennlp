# This Dockerfile creates an environment suitable for downstream usage of AllenNLP.
# It creates an environment that includes a pip installation of allennlp.

FROM python:3.6.8-stretch

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

ARG VERSION
ARG SOURCE_COMMIT

# Install the specified version of AllenNLP.
RUN if [ ! -z "$VERSION" ]; \
    then echo "Installing allennlp==$VERSION."; pip install allennlp==$VERSION; \
    elif [ ! -z "$SOURCE_COMMIT" ]; \
    then echo "Installing allennlp@$SOURCE_COMMIT"; pip install "git+git://github.com/allenai/allennlp.git@$SOURCE_COMMIT"; \
    else echo "Installing the latest pip release of allennlp"; pip install allennlp; \
    fi

LABEL maintainer="allennlp-contact@allenai.org"

ENV ALLENNLP_VERSION=$VERSION
ENV ALLENNLP_SOURCE_COMMIT=$SOURCE_COMMIT

ENTRYPOINT ["allennlp"]
