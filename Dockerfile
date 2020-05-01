# This Dockerfile creates an environment suitable for downstream usage of AllenNLP.
# It's built from a wheel installation of allennlp to ensure that the source matches
# what you'd get from a PyPI install.

FROM python:3.7

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

# Installing AllenNLP's dependencies is the most time-consuming part of building
# this Docker image, so we make use of layer caching here by adding the minimal files
# necessary to install the dependencies. Since most of the dependencies are defined
# in the setup.py file, we create a "shell" package of allennlp using the same setup file
# and then pip install it, after which we uninstall it so that we'll only have the dependencies
# installed.
COPY allennlp/version.py allennlp/version.py
RUN touch allennlp/__init__.py && touch README.md
COPY setup.py setup.py
COPY dev-requirements.txt dev-requirements.txt

# Now install deps by installing the shell package, and then uninstall it so we can
# re-install the full package below.
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r dev-requirements.txt && \
    pip uninstall -y typing && \
    pip uninstall -y allennlp && \
    rm -rf allennlp/

# Now add and install the wheel of AllenNLP.
COPY dist dist/
RUN pip install $(ls dist/*.whl)

# Copy wrapper script to allow beaker to run resumable training workloads.
COPY scripts/ai2_internal/resumable_train.sh /stage/allennlp

LABEL maintainer="allennlp-contact@allenai.org"

ENTRYPOINT ["allennlp"]
