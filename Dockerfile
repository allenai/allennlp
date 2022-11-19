# This Dockerfile creates an environment suitable for downstream usage of AllenNLP.
# It's built from a wheel installation of allennlp using the base images from
# https://github.com/allenai/docker-images/pkgs/container/pytorch

ARG TORCH=1.12.0-cuda11.3-python3.8
FROM ghcr.io/allenai/pytorch:${TORCH}

WORKDIR /stage/allennlp

# Installing AllenNLP's dependencies is the most time-consuming part of building
# this Docker image, so we make use of layer caching here by adding the minimal files
# necessary to install the dependencies.
COPY allennlp/version.py allennlp/version.py
COPY setup.py .
COPY requirements.txt .
COPY dev-requirements.txt .
COPY constraints.txt .
RUN touch allennlp/__init__.py \
    && touch README.md \
    && pip install --no-cache-dir -c constraints.txt -e .[all]

# Now add the full package source and re-install just the package.
COPY allennlp allennlp
RUN pip install --no-cache-dir --no-deps -e .[all]

COPY Makefile .
RUN make download-extras

WORKDIR /app/

# Copy wrapper script to allow beaker to run resumable training workloads.
COPY scripts/ai2_internal/resumable_train.sh .

LABEL maintainer="allennlp-contact@allenai.org"

ENTRYPOINT ["allennlp"]
