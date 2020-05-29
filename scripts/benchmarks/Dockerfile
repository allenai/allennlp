# Used to build an image for running benchmarks.
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

ENTRYPOINT ["allennlp"]

# Should specify which version allennlp to install. For example,
#  - ALLENNLP='allennlp==1.0.0rc5', or
#  - ALLENNLP='git+https://github.com/allenai/allennlp.git@master'
ARG ALLENNLP

RUN pip install --no-cache-dir ${ALLENNLP}

# Should specify which version allennlp-models to install. For example,
#  - MODELS='allennlp-models==1.0.0rc5', or
#  - MODELS='git+https://github.com/allenai/allennlp-models.git@master'
ARG MODELS

# Ensure allennlp isn't re-installed when we install allennlp-models.
ENV ALLENNLP_VERSION_OVERRIDE allennlp

# To be compatible with older versions of allennlp-models.
ENV IGNORE_ALLENNLP_IN_SETUP true

RUN pip install --no-cache-dir ${MODELS}
