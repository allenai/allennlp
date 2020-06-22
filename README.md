<div align="center">
    <br>
    <img src="https://raw.githubusercontent.com/allenai/allennlp/master/docs/img/allennlp-logo-dark.png" width="400"/>
    <p>
    An Apache 2.0 NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
    </p>
    <hr/>
</div>
<p align="center">
    <a href="https://github.com/allenai/allennlp/actions">
        <img alt="Build" src="https://github.com/allenai/allennlp/workflows/Master/badge.svg?event=push&branch=master">
    </a>
    <a href="https://pypi.org/project/allennlp/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/allennlp">
    </a>
    <a href="https://github.com/allenai/allennlp/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/allenai/allennlp.svg?color=blue&cachedrop">
    </a>
    <a href="https://codecov.io/gh/allenai/allennlp">
        <img alt="Codecov" src="https://codecov.io/gh/allenai/allennlp/branch/master/graph/badge.svg">
    </a>
    <a href="https://optuna.org">
        <img alt="Optuna" src="https://img.shields.io/badge/Optuna-integrated-blue">
    </a>
<br/>

## Quick Links

- [Website](https://allennlp.org/)
- [Guide](https://guide.allennlp.org/)
- [Forum](https://discourse.allennlp.org)
- Documentation ( [latest](https://docs.allennlp.org/latest/) | [stable](https://docs.allennlp.org/stable/) | [master](https://docs.allennlp.org/master/) )
- [Contributing Guidelines](CONTRIBUTING.md)
- [Pretrained Models](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/pretrained.py)
- [Continuous Build](https://github.com/allenai/allennlp/actions)
- [Nightly Releases](https://pypi.org/project/allennlp/#history)
- [Officially Supported Models](https://github.com/allenai/allennlp-models)

## Getting Started Using the Library

If you're interested in using AllenNLP for model development, we recommend you check out the
[AllenNLP Guide](https://guide.allennlp.org).  When you're ready to start your project, we've
created a couple of template repositories that you can use as a starting place:

* If you want to use `allennlp train` and config files to specify experiments, use [this
  template](https://github.com/allenai/allennlp-template-config-files). We recommend this approach.
* If you'd prefer to use python code to configure your experiments and run your training loop, use
  [this template](https://github.com/allenai/allennlp-template-python-script). There are a few
  things that are currently a little harder in this setup (loading a saved model, and using
  distributed training), but except for those its functionality is equivalent to the config files
  setup.

In addition, there are external tutorials:

* [Hyperparameter optimization for AllenNLP using Optuna](https://medium.com/optuna/hyperparameter-optimization-for-allennlp-using-optuna-54b4bfecd78b)

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
    <td><b> allennlp.training </b></td>
    <td> functionality for training models </td>
</tr>
</table>

## Installation

AllenNLP requires Python 3.6.1 or later. The preferred way to install AllenNLP is via `pip`.  Just run `pip install allennlp` in your Python environment and you're good to go!

If you need pointers on setting up an appropriate Python environment or would like to install AllenNLP using a different method, see below.

We support AllenNLP on Mac and Linux environments. We presently do not support Windows but are open to contributions.

### Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for AllenNLP.  If you already have a Python 3.6 or 3.7
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with Python 3.7:

    ```
    conda create -n allennlp python=3.7
    ```

3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use AllenNLP:

    ```
    conda activate allennlp
    ```

#### Installing the library and dependencies

Installing the library and dependencies is simple using `pip`.

```bash
pip install allennlp
```

*Looking for bleeding edge features? You can install nightly releases directly from [pypi](https://pypi.org/project/allennlp/#history)*

AllenNLP installs a script when you install the python package, so you can run allennlp commands just by typing `allennlp` into a terminal.  For example, you can now test your installation with `allennlp test-install`.

You may also want to install `allennlp-models`, which contains the NLP constructs to train and run our officially
supported models, many of which are hosted at [https://demo.allennlp.org](https://demo.allennlp.org).

```bash
pip install allennlp-models
```

### Installing using Docker

Docker provides a virtual machine with everything set up to run AllenNLP--
whether you will leverage a GPU or just run on a CPU.  Docker provides more
isolation and consistency, and also makes it easy to distribute your
environment to a compute cluster.

Once you have [installed Docker](https://docs.docker.com/engine/installation/)
just run the following command to get an environment that will run on either the cpu or gpu.

```bash
mkdir -p $HOME/.allennlp/
docker run --rm -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:latest
```

You can test the Docker environment with

```bash
docker run --rm -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:latest test-install 
```

### Installing from source

You can also install AllenNLP by cloning our git repository:

```bash
git clone https://github.com/allenai/allennlp.git
```

Create a Python 3.7 virtual environment, and install AllenNLP in `editable` mode by running:

```bash
pip install --editable .
pip install -r dev-requirements.txt
```

This will make `allennlp` available on your system but it will use the sources from the local clone
you made of the source repository.

You can test your installation with `allennlp test-install`.
See [https://github.com/allenai/allennlp-models](https://github.com/allenai/allennlp-models)
for instructions on installing `allennlp-models` from source.

## Running AllenNLP

Once you've installed AllenNLP, you can run the command-line interface
with the `allennlp` command (whether you installed from `pip` or from source).
`allennlp` has various subcommands such as `train`, `evaluate`, and `predict`.
To see the full usage information, run `allennlp --help`.

## Docker images

AllenNLP releases Docker images to [Docker Hub](https://hub.docker.com/r/allennlp/) for each release.  For information on how to run these releases, see [Installing using Docker](#installing-using-docker).

### Building a Docker image

For various reasons you may need to create your own AllenNLP Docker image.
The same image can be used either with a CPU or a GPU.

First, you need to [install Docker](https://www.docker.com/get-started).
Then you will need a wheel of allennlp in the `dist/` directory.
You can either obtain a pre-built wheel from a PyPI release or build a new wheel from
source.

PyPI release wheels can be downloaded by going to https://pypi.org/project/allennlp/#history,
clicking on the desired release, and then clicking "Download files" in the left sidebar.
After downloading, make you sure you put the wheel in the `dist/` directory
(which may not exist if you haven't built a wheel from source yet).

To build a wheel from source, just run `python setup.py wheel`.

*Before building the image, make sure you only have one wheel in the `dist/` directory.*

Once you have your wheel, run `make docker-image`. By default this builds an image
with the tag `allennlp/allennlp`. You can change this to anything you want
by setting the `DOCKER_TAG` flag when you call `make`. For example,
`make docker-image DOCKER_TAG=my-allennlp`.

You should now be able to see this image listed by running `docker images allennlp`.

```
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
allennlp/allennlp   latest              b66aee6cb593        5 minutes ago       2.38GB
```

### Running the Docker image

You can run the image with `docker run --rm -it allennlp/allennlp:latest`.  The `--rm` flag cleans up the image on exit and the `-it` flags make the session interactive so you can use the bash shell the Docker image starts.

You can test your installation by running  `allennlp test-install`.

## Issues

Everyone is welcome to file issues with either feature requests, bug reports, or general questions.  As a small team with our own internal goals, we may ask for contributions if a prompt fix doesn't fit into our roadmap.  To keep things tidy we will often close issues we think are answered, but don't hesitate to follow up if further discussion is needed.

## Contributions

The AllenNLP team at AI2 (@allenai) welcomes contributions from the greater AllenNLP community, and, if you would like to get a change into the library, this is likely the fastest approach.  If you would like to contribute a larger feature, we recommend first creating an issue with a proposed design for discussion.  This will prevent you from spending significant time on an implementation which has a technical limitation someone could have pointed out early on.  Small contributions can be made directly in a pull request.

Pull requests (PRs) must have one approving review and no requested changes before they are merged.  As AllenNLP is primarily driven by AI2 (@allenai) we reserve the right to reject or revert contributions that we don't think are good additions.

## Citing

If you use AllenNLP in your research, please cite [AllenNLP: A Deep Semantic Natural Language Processing Platform](https://www.semanticscholar.org/paper/AllenNLP%3A-A-Deep-Semantic-Natural-Language-Platform-Gardner-Grus/a5502187140cdd98d76ae711973dbcdaf1fef46d).

```bibtex
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

AllenNLP is an open-source project backed by [the Allen Institute for Artificial Intelligence (AI2)](https://allenai.org/).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/allennlp/graphs/contributors) page.
