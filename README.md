[![Build Status](https://travis-ci.org/allenai/allennlp.svg?branch=master)](https://travis-ci.org/allenai/allennlp)
[![codecov](https://codecov.io/gh/allenai/allennlp/branch/master/graph/badge.svg)](https://codecov.io/gh/allenai/allennlp)
# AllenNLP

A [Apache 2.0](https://github.com/allenai/allennlp/blob/master/LICENSE) natural language processing toolkit using state-of-the-art deep learning models.

## Running AllenNLP

### Setting up a Conda development environment

[Conda](https://conda.io/) will set up a virtual environment with the exact version of Python
used for development along with all the dependencies needed to run AllenNLP.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.

    ```
    conda create -n allennlp python=3.5
    ```

3.  Now activate the Conda environment.

    ```
    source activate allennlp
    ```

4.  Install the required dependencies.

    ```
    INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh
    ```

5. Visit http://pytorch.org/ and install the relevant pytorch package.

6.  Set the `PYTHONHASHSEED` for repeatable experiments.

    ```
    export PYTHONHASHSEED=2157
    ```

You should now be able to test your installation with `pytest -v`.  Congratulations!

### Setting up a Docker development environment

Docker provides a virtual machine with everything set up to run AllenNLP--whether you will leverage a GPU or just
run on a CPU.  Docker provides more isolation and consistency, and also makes it easy to distribute your environment
to a compute cluster.

## Downloading a pre-built Docker image

It is easy to run a pre-built Docker development environment.  AllenNLP is configured with Docker Cloud to build a
new image on every update to the master branch.  To download an image from [Docker Hub](https://hub.docker.com/r/allennlp/):

```bash
docker pull allennlp/allennlp-cpu:latest
```

You can alternatively download an environment set up to use a GPU.

```bash
docker pull allennlp/allennlp-gpu:latest
```

## Building a Docker image

Following are instructions on creating a Docker environment that use the CPU.  To use the GPU, use the same instructions
but substitute `gpu` for `cpu`.  The following command will take some time, as it completely builds the environment
needed to run AllenNLP.

```bash
docker build --file Dockerfile.cpu --tag allennlp/allennlp-cpu .
```

You should now be able to see this image listed by running `docker images allennlp-cpu`.

```
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
allennlp/allennlp-cpu        latest              b66aee6cb593        5 minutes ago       2.38GB
```

## Running the Docker image

You can run the image with `docker run --rm -it allennlp/allennlp-cpu`.  The `--rm` flag cleans up the image on exit and the
`-it` flags make the session interactive so you can use the bash shell the Docker image starts.

The Docker environment uses Conda to install Python and automatically enters the Conda environment "allennlp".

You can test your installation by running  `pytest -v`.


### Setting up a Kubernetes development environment

Kubernetes will deploy your Docker images into the cloud, so you can have a reproducible development environment on AWS.

1. Follow the instructions for getting started with
[Kubernetes](https://github.com/allenai/infrastructure/tree/master/kubernetes).

2. Run `kubectl create -f /path/to/kubernetes-dev-environment.yaml`.  This will create a "job" on the cluster which you
can later connect to using bash.  Note that you will be using the last Dockerfile that would pushed, and so the source
code may not match what you have locally.

4. Retrieve the name of the pod created with `kubectl describe job <JOBNAME> --namespace=allennlp`.
The pod name will be your job name followed by some additional characters.

5. Get a shell inside the container using `kubectl exec -it <PODNAME> bash`

6. When you are done, don't forget to kill your job using `kubectl delete -f /path/to/kubernetes-dev-environment.yaml`

