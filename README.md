<div align="center">
    <br>
    <img src="https://raw.githubusercontent.com/allenai/allennlp/main/docs/img/allennlp-logo-dark.png" width="400"/>
    <p>
    An Apache 2.0 NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks.
    </p>
    <hr/>
</div>
<p align="center">
    <a href="https://github.com/allenai/allennlp/actions">
        <img alt="CI" src="https://github.com/allenai/allennlp/workflows/CI/badge.svg?event=push&branch=main">
    </a>
    <a href="https://pypi.org/project/allennlp/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/allennlp">
    </a>
    <a href="https://github.com/allenai/allennlp/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/allenai/allennlp.svg?color=blue&cachedrop">
    </a>
    <a href="https://codecov.io/gh/allenai/allennlp">
        <img alt="Codecov" src="https://codecov.io/gh/allenai/allennlp/branch/main/graph/badge.svg">
    </a>
    <a href="https://optuna.org">
        <img alt="Optuna" src="https://img.shields.io/badge/Optuna-integrated-blue">
    </a>
    <br/>
</p>

## Quick Links

- [Website](https://allennlp.org/)
- [Guide](https://guide.allennlp.org/)
- [Documentation](https://docs.allennlp.org/) ( [latest](https://docs.allennlp.org/latest/) | [stable](https://docs.allennlp.org/stable/) | [commit](https://docs.allennlp.org/main/) )
- [Forum](https://discourse.allennlp.org)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/allennlp)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Officially Supported Models](https://github.com/allenai/allennlp-models)
    - [Pretrained Models](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/pretrained.py)
    - [Documentation](https://docs.allennlp.org/models/) ( [latest](https://docs.allennlp.org/models/latest/) | [stable](https://docs.allennlp.org/models/stable/) | [commit](https://docs.allennlp.org/models/main/) )
- [Continuous Build](https://github.com/allenai/allennlp/actions)
- [Nightly Releases](https://pypi.org/project/allennlp/#history)

## Getting Started Using the Library

If you're interested in using AllenNLP for model development, we recommend you check out the
[AllenNLP Guide](https://guide.allennlp.org).  When you're ready to start your project, we've
created a couple of template repositories that you can use as a starting place:

* If you want to use `allennlp train` and config files to specify experiments, use [this
  template](https://github.com/allenai/allennlp-template-config-files). We recommend this approach.
* If you'd prefer to use python code to configure your experiments and run your training loop, use
  [this template](https://github.com/allenai/allennlp-template-python-script). There are a few
  things that are currently a little harder in this setup (loading a saved model, and using
  distributed training), but otherwise it's functionality equivalent to the config files
  setup.

In addition, there are external tutorials:

* [Hyperparameter optimization for AllenNLP using Optuna](https://medium.com/optuna/hyperparameter-optimization-for-allennlp-using-optuna-54b4bfecd78b)
* [Training with multiple GPUs in AllenNLP](https://medium.com/ai2-blog/tutorial-how-to-train-with-multiple-gpus-in-allennlp-c4d7c17eb6d6)

And others on the [AI2 AllenNLP blog](https://medium.com/ai2-blog/allennlp/home).

## Plugins

AllenNLP supports loading "plugins" dynamically. A plugin is just a Python package that
provides custom registered classes or additional `allennlp` subcommands.

There is ecosystem of open source plugins, some of which are maintained by the AllenNLP
team here at AI2, and some of which are maintained by the broader community.

<table>
<tr>
    <td><b> Plugin </b></td>
    <td><b> Maintainer </b></td>
    <td><b> CLI </b></td>
    <td><b> Description </b></td>
</tr>
<tr>
    <td> <a href="https://github.com/allenai/allennlp-models"><b>allennlp-models</b></a> </td>
    <td> AI2 </td>
    <td> No </td>
    <td> A collection of state-of-the-art models </td>
</tr>
<tr>
    <td> <a href="https://github.com/allenai/allennlp-semparse"><b>allennlp-semparse</b></a> </td>
    <td> AI2 </td>
    <td> No </td>
    <td> A framework for building semantic parsers </td>
</tr>
<tr>
    <td> <a href="https://github.com/allenai/allennlp-server"><b>allennlp-server</b></a> </td>
    <td> AI2 </td>
    <td> Yes </td>
    <td> A simple demo server for serving models </td>
</tr>
<tr>
    <td> <a href="https://github.com/himkt/allennlp-optuna"><b>allennlp-optuna</b></a> </td>
    <td> <a href="https://himkt.github.io/profile/">Makoto Hiramatsu</a> </td>
    <td> Yes </td>
    <td> <a href="https://optuna.org/">Optuna</a> integration for hyperparameter optimization </td>
</tr>
</table>

AllenNLP will automatically find any official AI2-maintained plugins that you have installed,
but for AllenNLP to find personal or third-party plugins you've installed,
you also have to create either a local plugins file named `.allennlp_plugins`
in the directory where you run the `allennlp` command, or a global plugins file at `~/.allennlp/plugins`.
The file should list the plugin modules that you want to be loaded, one per line.

To test that your plugins can be found and imported by AllenNLP, you can run the `allennlp test-install` command.
Each discovered plugin will be logged to the terminal.

For more information about plugins, see the [plugins API docs](https://docs.allennlp.org/main/api/common/plugins/). And for information on how to create a custom subcommand
to distribute as a plugin, see the [subcommand API docs](https://docs.allennlp.org/main/api/commands/subcommand/).

## Package Overview

<table>
<tr>
    <td><b> allennlp </b></td>
    <td> An open-source NLP research library, built on PyTorch </td>
</tr>
<tr>
    <td><b> allennlp.commands </b></td>
    <td> Functionality for the CLI </td>
</tr>
<tr>
    <td><b> allennlp.common </b></td>
    <td> Utility modules that are used across the library </td>
</tr>
<tr>
    <td><b> allennlp.data </b></td>
    <td> A data processing module for loading datasets and encoding strings as integers for representation in matrices </td>
</tr>
<tr>
    <td><b> allennlp.modules </b></td>
    <td> A collection of PyTorch modules for use with text </td>
</tr>
<tr>
    <td><b> allennlp.nn </b></td>
    <td> Tensor utility functions, such as initializers and activation functions </td>
</tr>
<tr>
    <td><b> allennlp.training </b></td>
    <td> Functionality for training models </td>
</tr>
</table>

## Installation

AllenNLP requires Python 3.6.1 or later and [PyTorch](https://pytorch.org/).
It's recommended that you install the PyTorch ecosystem **before** installing AllenNLP by following the instructions on [pytorch.org](https://pytorch.org/).

The preferred way to install AllenNLP is via `pip`. Just run `pip install allennlp`.

> ⚠️ If you're using Python 3.7 or greater, you should ensure that you don't have the PyPI version of `dataclasses` installed after running the above command, as this could cause issues on certain platforms. You can quickly check this by running `pip freeze | grep dataclasses`. If you see something like `dataclasses=0.6` in the output, then just run `pip uninstall -y dataclasses`.

If you need pointers on setting up an appropriate Python environment or would like to install AllenNLP using a different method, see below.

We support AllenNLP on Mac and Linux environments. We presently do not support Windows but are open to contributions.

### Installing via pip

#### Setting up a virtual environment

[Conda](https://conda.io/) can be used set up a virtual environment with the
version of Python required for AllenNLP.  If you already have a Python 3
environment you want to use, you can skip to the 'installing via pip' section.

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with Python 3.7 (3.6 or 3.8 would work as well):

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

AllenNLP provides [official Docker images](https://hub.docker.com/r/allennlp/allennlp) with the library and all of its dependencies installed.

Once you have [installed Docker](https://docs.docker.com/engine/installation/),
you should also install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
if you have GPUs available.

Then run the following command to get an environment that will run on GPU:

```bash
mkdir -p $HOME/.allennlp/
docker run --rm --gpus all -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:latest
```

You can test the Docker environment with

```bash
docker run --rm --gpus all -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:latest test-install 
```

If you don't have GPUs available, just omit the `--gpus all` flag.

#### Building your own Docker image

For various reasons you may need to create your own AllenNLP Docker image, such as if you need a different version
of PyTorch. To do so, just run `make docker-image` from the root of your local clone of AllenNLP.

By default this builds an image with the tag `allennlp/allennlp`, but you can change this to anything you want
by setting the `DOCKER_TAG` flag when you call `make`. For example,
`make docker-image DOCKER_TAG=my-allennlp`.

If you want to use a different version of PyTorch, set the flag `DOCKER_TORCH_VERSION` to something like
`torch==1.7.0` or `torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html`.
The value of this flag will passed directly to `pip install`.

After building the image you should be able to see it listed by running `docker images allennlp`.

```
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
allennlp/allennlp   latest              b66aee6cb593        5 minutes ago       2.38GB
```

### Installing from source

You can also install AllenNLP by cloning our git repository:

```bash
git clone https://github.com/allenai/allennlp.git
```

Create a Python 3.7 or 3.8 virtual environment, and install AllenNLP in `editable` mode by running:

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
