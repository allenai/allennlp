FROM 896129387501.dkr.ecr.us-west-2.amazonaws.com/deep_qa/cuda:8

ENTRYPOINT ["/bin/bash", "-c"]

WORKDIR /stage

RUN conda create -n runenv --yes python=3.5
ENV PYTHONHASHSEED 2157

COPY . .

RUN /bin/bash -c "source activate runenv && scripts/install_requirements.sh"

CMD ["/bin/sh", "-c]
