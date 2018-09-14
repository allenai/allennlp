<p align="center"><img width="40%" src="doc/static/allennlp-logo-dark.png" /></p>

[![Build Status](http://build.allennlp.org/app/rest/builds/buildType:(id:AllenNLP_AllenNLPCommits)/statusIcon)](http://build.allennlp.org/viewType.html?buildTypeId=AllenNLP_AllenNLPCommits&guest=1)
[![codecov](https://codecov.io/gh/allenai/allennlp/branch/master/graph/badge.svg)](https://codecov.io/gh/allenai/allennlp)
[![docker](https://images.microbadger.com/badges/version/allennlp/allennlp.svg)](https://microbadger.com/images/allennlp/allennlp)

An [Apache 2.0](https://github.com/allenai/allennlp/blob/master/LICENSE) NLP research library, built on PyTorch,
for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.

## Installation

AllenNLP requires Python 3.6.1 or later. The preferred way to install AllenNLP is via `pip`.  Just run `pip install allennlp` in your Python environment and you're good to go!

If you need pointers on setting up an appropriate Python environment or would like to install AllenNLP using a different method, see below.

### Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for AllenNLP.  If you already have a Python 3.6 or 3.7
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```bash
    conda create -n allennlp python=3.6
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use AllenNLP.

    ```bash
    source activate allennlp
    ```

#### Installing the library and dependencies

Installing the library and dependencies is simple using `pip`.

   ```bash
   pip install allennlp
   ```

That's it! You're now ready to build and train AllenNLP models.
AllenNLP installs a script when you install the python package, meaning you can run allennlp commands just by typing `allennlp` into a terminal.

You can now test your installation with `./scripts/verify.py`.

_`pip` currently installs Pytorch for CUDA 9 only (or no GPU). If you require an older version,
please visit http://pytorch.org/ and install the relevant pytorch binary._

### Installing using Docker

Docker provides a virtual machine with everything set up to run AllenNLP--
whether you will leverage a GPU or just run on a CPU.  Docker provides more
isolation and consistency, and also makes it easy to distribute your
environment to a compute cluster.

Once you have [installed Docker](https://docs.docker.com/engine/installation/)
just run the following command to get an environment that will run on either the cpu or gpu.

   ```bash
   docker run -it -p 8000:8000 --rm allennlp/allennlp:v0.6.1` 
   ```

You can now test your installation with `./scripts/verify.py`.

### Installing from source

You can also install AllenNLP by cloning our git repository:

  ```bash
  git clone https://github.com/allenai/allennlp.git
  ```

Create a Python 3.6 virtual environment, and install the necessary requirements by running:

  ```bash
  INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
  ```

Changing the flag to false if you don't want to be able to run
tests. Once the requirements have been installed, run:

  ```bash
  pip install --editable .
  ```

To install the AllenNLP library in `editable` mode into your
environment.  This will make `allennlp` available on your
system but it will use the sources from the local clone you
made of the source repository.

You can test your installation with `./scripts/verify.py`.

## Running AllenNLP

Once you've installed AllenNLP, you can run the command-line interface either
with the `allennlp` command (if you installed via `pip`) or `bin/allennlp` (if you installed via source).

```bash
$ allennlp
Run AllenNLP

optional arguments:
  -h, --help    show this help message and exit
  --version     show program's version number and exit

Commands:
  
    configure   Generate configuration stubs.
    train       Train a model
    evaluate    Evaluate the specified model + dataset
    predict     Use a trained model to make predictions.
    make-vocab  Create a vocabulary
    elmo        Use a trained model to make predictions.
    fine-tune   Continue training a model on a new dataset
    dry-run     Create a vocabulary, compute dataset statistics and other
                training utilities.
    test-install
                Run the unit tests.
```

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
attention), and more (see http://www.allennlp.org/models).

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
    <td> a web server to that can serve demos for your models </td>
</tr>
<tr>
    <td><b> allennlp.training </b></td>
    <td> functionality for training models </td>
</tr>
</table>

## Docker images

AllenNLP releases Docker images to [Docker Hub](https://hub.docker.com/r/allennlp/) for each release.  For information on how to run these releases, see [Installing using Docker](#installing-using-docker).

### Building a Docker image

For various reasons you may need to create your own AllenNLP Docker image.
The same image can be used either with a CPU or a GPU.

First, you need to [install Docker](https://www.docker.com/get-started).
Then run the following command
(it will take some time, as it completely builds the
environment needed to run AllenNLP.)

```bash
docker build -f Dockerfile.pip --tag allennlp/allennlp:latest .
```

You should now be able to see this image listed by running `docker images allennlp`.

```
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
allennlp/allennlp            latest              b66aee6cb593        5 minutes ago       2.38GB
```

### Running the Docker image

You can run the image with `docker run --rm -it allennlp/allennlp:latest`.  The `--rm` flag cleans up the image on exit and the `-it` flags make the session interactive so you can use the bash shell the Docker image starts.

You can test your installation by running  `./scripts/verify.py`.

## Citing

If you use AllenNLP in your research, please cite [AllenNLP: A Deep Semantic Natural Language Processing Platform](https://www.semanticscholar.org/paper/AllenNLP%3A-A-Deep-Semantic-Natural-Language-Platform-Gardner-Grus/a5502187140cdd98d76ae711973dbcdaf1fef46d).

```
@inproceedings{Gardner2017AllenNLP,
  title={AllenNLP: A Deep Semantic Natural Language Processing Platform},
  author={Matt Gardner and Joel Grus and Mark Neumann and Oyvind Tafjord
    and Pradeep Dasigi and Nelson F. Liu and Matthew Peters and
    Michael Schmitz and Luke S. Zettlemoyer},
  year={2017},
  Eprint = {arXiv:1803.07640},
}
```

## Team

AllenNLP is an open-source project backed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/allennlp/graphs/contributors) page.
