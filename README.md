<p align="center"><img width="40%" src="doc/static/allennlp-logo-dark.png" /></p>

[![Build Status](http://build.allennlp.org/app/rest/builds/buildType:(id:AllenNLP_AllenNLPCommits)/statusIcon)](http://build.allennlp.org/viewType.html?buildTypeId=AllenNLP_AllenNLPCommits&guest=1)
[![codecov](https://codecov.io/gh/allenai/allennlp/branch/master/graph/badge.svg)](https://codecov.io/gh/allenai/allennlp)

An [Apache 2.0](https://github.com/allenai/allennlp/blob/master/LICENSE) NLP research library, built on PyTorch,
for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.

## Quick Links

* [Website](http://www.allennlp.org/)
* [Tutorial](https://allennlp.org/tutorials)
* [Documentation](https://allenai.github.io/allennlp-docs/)
* [Contributing Guidelines](CONTRIBUTING.md)
* [Model List](MODELS.md)
* [Continuous Build](http://build.allennlp.org/)

## Package Overview

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

## Installation

AllenNLP requires Python 3.6.1 or later. The preferred way to install AllenNLP is via `pip`.  Just run `pip install allennlp` in your Python environment and you're good to go!

If you need pointers on setting up an appropriate Python environment or would like to install AllenNLP using a different method, see below.

Windows is currently not officially supported, although we try to fix issues when they are easily addressed.

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

You can now test your installation with `allennlp test-install`.

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
   mkdir -p $HOME/.allennlp/
   docker run --rm -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:v0.8.3
   ```

You can test the Docker environment with `docker run --rm -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:v0.8.3 test-install`.

### Installing from source

You can also install AllenNLP by cloning our git repository:

  ```bash
  git clone https://github.com/allenai/allennlp.git
  ```

Create a Python 3.6 virtual environment, and install AllenNLP in `editable` mode by running:

  ```bash
  pip install --editable .
  ```

This will make `allennlp` available on your system but it will use the sources from the local clone
you made of the source repository.

You can test your installation with `allennlp test-install`.
The full development environment also requires the JVM and `perl`,
which must be installed separately.  `./scripts/verify.py` will run
the full suite of tests used by our continuous build environment.

## Running AllenNLP

Once you've installed AllenNLP, you can run the command-line interface either
with the `allennlp` command (if you installed via `pip`) or `allennlp` (if you installed via source).

```bash
$ allennlp
Run AllenNLP

optional arguments:
  -h, --help    show this help message and exit
  --version     show program's version number and exit

Commands:

    configure   Run the configuration wizard.
    train       Train a model.
    evaluate    Evaluate the specified model + dataset.
    predict     Use a trained model to make predictions.
    make-vocab  Create a vocabulary.
    elmo        Create word vectors using a pretrained ELMo model.
    fine-tune   Continue training a model on a new dataset.
    dry-run     Create a vocabulary, compute dataset statistics and other
                training utilities.
    test-install
                Run the unit tests.
    find-lr     Find a learning rate range.
```

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

You can test your installation by running  `allennlp test-install`.

## Issues

Everyone is welcome to file issues with either feature requests, bug reports, or general questions.  As a small team with our own internal goals, we may ask for contributions if a prompt fix doesn't fit into our roadmap.  We allow users a two week window to follow up on questions, after which we will close issues.  They can be re-opened if there is further discussion.

## Contributions

The AllenNLP team at AI2 (@allenai) welcomes contributions from the greater AllenNLP community, and, if you would like to get a change into the library, this is likely the fastest approach.  If you would like to contribute a larger feature, we recommend first creating an issue with a proposed design for discussion.  This will prevent you from spending significant time on an implementation which has a technical limitation someone could have pointed out early on.  Small contributions can be made directly in a pull request.

Pull requests (PRs) must have one approving review and no requested changes before they are merged.  As AllenNLP is primarily driven by AI2 (@allenai) we reserve the right to reject or revert contributions that we don't think are good additions.

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
