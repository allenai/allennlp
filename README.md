<p align="center"><img width="40%" src="doc/static/allennlp-logo-dark.png" /></p>

[![Build Status](http://build.allennlp.org/app/rest/builds/buildType:(id:AllenNLP_AllenNLPCommits)/statusIcon)](http://build.allennlp.org/viewType.html?buildTypeId=AllenNLP_AllenNLPCommits&guest=1)
[![codecov](https://codecov.io/gh/allenai/allennlp/branch/master/graph/badge.svg)](https://codecov.io/gh/allenai/allennlp)
[![docker](https://images.microbadger.com/badges/version/allennlp/allennlp.svg)](https://microbadger.com/images/allennlp/allennlp)

An [Apache 2.0](https://github.com/allenai/allennlp/blob/master/LICENSE) NLP research library, built on PyTorch,
for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.

## Quickstart

The fastest way to get an environment to run AllenNLP is with Docker.  Once you have [installed Docker](https://docs.docker.com/engine/installation/)
just run `docker run -it --rm allennlp/allennlp:v0.3.0` to get an environment that will run on either the cpu or gpu.

Now you can do any of the following:

* Run a model on example sentences with `python -m allennlp.run predict`.
* Start a web service to host our models with `python -m allennlp.run serve`.
* Interactively code against AllenNLP from the Python interpreter with `python`.

You can also install via the `pip` package manager or by cloning this repository into a Python 3.6 environment.
See below for more detailed instructions.

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

### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment
with the version of Python required for AllenNLP and in which you can
sandbox its dependencies:

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```
    conda create -n allennlp python=3.6
    ```

3.  Activate the Conda environment.  (You will need to activate the Conda environment in each terminal in which you want to use AllenNLP.

    ```
    source activate allennlp
    ```

4. Install AllenNLP in your environment.

### Installing via pip

The preferred way to install AllenNLP into your environment is via `pip`:

```
pip install allennlp
```

You will also need to manually install some dependencies:

1. Visit http://pytorch.org/ and install the relevant pytorch package.

2. Download necessary spacy models. `python -m spacy download en_core_web_sm`.

That's it! You're now ready to build and train AllenNLP models.

### Setting up a development environment

If you want to make changes to AllenNLP library itself
(or use bleeding-edge code that hasn't been released to PyPI)
you'll need to install the library from GitHub and manually install the requirements:

1. First, clone the repo:

```
git clone https://github.com/allenai/allennlp.git
```

2. Change your directory to where you cloned the files:

```
cd allennlp
```

3.  Install the required dependencies.

    ```
    INSTALL_TEST_REQUIREMENTS="true" ./scripts/install_requirements.sh
    ```

4. Visit http://pytorch.org/ and install the relevant pytorch package.

You should now be able to test your installation with `./scripts/verify.py`.  Congratulations!

### Setting up a Docker development environment

A third option is to run AllenNLP via Docker.
Docker provides a virtual machine with everything set up to run AllenNLP--
whether you will leverage a GPU or just run on a CPU.
Docker provides more isolation and consistency,
and also makes it easy to distribute your environment
to a compute cluster.

#### Downloading a pre-built Docker image

It is easy to run a pre-built Docker development environment.
AllenNLP is configured with Docker Cloud to build a
new image on every update to the master branch.  To download
the latest released from [Docker Hub](https://hub.docker.com/r/allennlp/) just run:

```bash
docker pull allennlp/allennlp:v0.3.0
```

#### Building a Docker image

For various reasons you may need to create your own AllenNLP Docker image.
The same image can be used either with a CPU or a GPU.

First, follow the instructions above for setting up a development environment.
Then run the following command
(it will take some time, as it completely builds the
environment needed to run AllenNLP.)

```bash
docker build --tag allennlp/allennlp .
```

You should now be able to see this image listed by running `docker images allennlp`.

```
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
allennlp/allennlp            latest              b66aee6cb593        5 minutes ago       2.38GB
```

#### Running the Docker image

You can run the image with `docker run --rm -it allennlp/allennlp`.  The `--rm` flag cleans up the image on exit and the
`-it` flags make the session interactive so you can use the bash shell the Docker image starts.

You can test your installation by running  `./scripts/verify.py`.


## Team

AllenNLP is an open-source project backed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/allennlp/graphs/contributors) page.
