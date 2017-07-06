FROM 896129387501.dkr.ecr.us-west-2.amazonaws.com/deep_qa/cuda:8

ENTRYPOINT ["/bin/bash", "-c"]

WORKDIR /stage

RUN conda create -n runenv --yes python=3.5
ENV PYTHONHASHSEED 2157

COPY . .

RUN /bin/bash -c 'source activate runenv && scripts/install_requirements.sh &&\
 pip install --no-cache-dir -q http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl &&\
 conda install pytorch torchvision -c soumith -y -q'

CMD ["/bin/sh", "-c]
