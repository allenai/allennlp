<p align="center"><img width="40%" src="doc/static/allennlp-logo-dark.png" /></p>


[![Build Status](https://travis-ci.org/allenai/allennlp.svg?branch=master)](https://travis-ci.org/allenai/allennlp)
[![codecov](https://codecov.io/gh/allenai/allennlp/branch/master/graph/badge.svg)](https://codecov.io/gh/allenai/allennlp)
[![docs](https://readthedocs.org/projects/allennlp/badge/?version=latest)](https://readthedocs.org/projects/allennlp/)

An [Apache 2.0](https://github.com/allenai/allennlp/blob/master/LICENSE) NLP research library, built on PyTorch,
for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.

## Quickstart

The fastest way to get an environment to run AllenNLP is with Docker.  Once you have [installed Docker](https://docs.docker.com/engine/installation/)
just run `docker run -it --rm allennlp/allennlp` to get an environment that will run on either the cpu or gpu.

Now you can do any of the following:

* Run a model on example sentences with `allennlp/run bulk`.
* Start a web service to host our models with `allennlp/run serve`.
* Interactively code against AllenNLP from the Python interpreter with `python`.

## What is AllenNLP?

Built on PyTorch, AllenNLP makes it easy to design and evaluate new deep
learning models for nearly any NLP problem, along with the infrastructure to
easily run them in the cloud or on your laptop.  AllenNLP was designed with the
following principles:

* *Hyper-modular and lightweight.* Use the parts which you like seamlessly with PyTorch.
* *Extensively tested and easy to extend.* Test coverage is above 90% and the example
  models provide a template for contributions.
* *Take padding and masking seriously*, making it easy to implement correct
  models without the pain.
* *Experiment friendly.*  Run reproducible experiments from a json
  specification with comprehensive logging.

AllenNLP includes reference implementations of high quality models for Semantic
Role Labelling, Question and Answering (BiDAF), Entailment (decomposable
attention), and more.

AllenNLP is built and maintained by the Allen Institute for Artificial
Intelligence, in close collaboration with researchers at the University of
Washington and elsewhere. With a dedicated team of best-in-field researchers
and software engineers, the AllenNLP project is uniquely positioned to provide
state of the art models with high quality engineering.

<table>
<tr>
    <td><b> allennlp </b></td>
    <td> an open-source NLP research library, built on PyTorch </td>
</tr>
<tr>
    <td><b> allennlp.commands </b></td>
    <td> functionality for a CLI and web service </td>
</tr>
<tr>
    <td><b> allennlp.data </b></td>
    <td> a data processing module for loading datasets and encoding strings as integers for representation in matrices </td>
</tr>
<tr>
    <td><b> allennlp.models </b></td>
    <td> a collection of state-of-the-art models </td>
</tr>
<tr>
    <td><b> allennlp.modules </b></td>
    <td> a collection of PyTorch modules for use with text </td>
</tr>
<tr>
    <td><b> allennlp.nn </b></td>
    <td> tensor utility functions, such as initializers and activation functions </td>
</tr>
<tr>
    <td><b> allennlp.service </b></td>
    <td> a web server to serve our demo and API </td>
</tr>
<tr>
    <td><b> allennlp.training </b></td>
    <td> functionality for training models </td>
</tr>
</table>

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
docker pull allennlp/allennlp:latest
```

## Building a Docker image

Following are instructions on creating a Docker environment that works on a CPU
or GPU.  The following command will take some time, as it completely builds the
environment needed to run AllenNLP.

```bash
docker build --tag allennlp/allennlp .
```

You should now be able to see this image listed by running `docker images allennlp`.

```
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
allennlp/allennlp            latest              b66aee6cb593        5 minutes ago       2.38GB
```

## Running the Docker image

You can run the image with `docker run --rm -it allennlp/allennlp`.  The `--rm` flag cleans up the image on exit and the
`-it` flags make the session interactive so you can use the bash shell the Docker image starts.

The Docker environment uses Conda to install Python and automatically enters the Conda environment "allennlp".

You can test your installation by running  `pytest -v`.


### Setting up a Kubernetes development environment

Kubernetes will deploy your Docker images into the cloud, so you can have a reproducible development environment on AWS.

1. Set up `kubectl` to connect to your Kubernetes cluster.

2. Run `kubectl create -f /path/to/kubernetes-dev-environment.yaml`.  This will create a "job" on the cluster which you
can later connect to using bash.  Note that you will be using the last Dockerfile that would pushed, and so the source
code may not match what you have locally.

4. Retrieve the name of the pod created with `kubectl describe job <JOBNAME> --namespace=allennlp`.
The pod name will be your job name followed by some additional characters.

5. Get a shell inside the container using `kubectl exec -it <PODNAME> bash`

6. When you are done, don't forget to kill your job using `kubectl delete -f /path/to/kubernetes-dev-environment.yaml`

## Team

AllenNLP is an open-source project backed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/allennlp/graphs/contributors)
page.
