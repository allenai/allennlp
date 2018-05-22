# Installation and Getting Started

Welcome to AllenNLP!

## Installing using Docker

The easiest way to get started is using Docker. Assuming you have Docker installed, just run

```bash
$ docker run -p 8000:8000 -it --rm allennlp/allennlp
```

This will download the latest `allennlp` image to your machine
(unless you already have it),
start a Docker container, and launch an interactive shell.
It also exposes port 8000, which is where the demo server runs,
and shuts down the container when you exit the interactive shell.

## Installing using pip

You can install `allennlp` using pip in two easy steps.

1.  Create a Python 3.6 virtual environment.  For example, if you use Conda:

    ```
    $ conda create -n allennlp python=3.6
    $ source activate allennlp
    ```

2.  Install `allennlp` via pip.

    ```
    pip install allennlp
    ```

## Installing from source

A third alternative is to clone from our git repository:

```bash
$ git clone https://github.com/allenai/allennlp.git
```

Create a Python 3.6 virtual environment, and install the necessary requirements
by running:

```bash
$ INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
```

changing the flag to `false` if you don't want to be able to run tests.
(Narrator: You want to be able to run tests.)

## Once You've Installed

If you just want to use the models and helper classes that are included with AllenNLP,
you can use the included `allennlp` command, which provides a command-line interface to
common functionality around training and evaluating models.  Note that if you are using
the source repository, you need to use `python -m allennlp.run` instead of `allennlp`.

```
Run AllenNLP

optional arguments:
  -h, --help    show this help message and exit

Commands:

    train       Train a model
    evaluate    Evaluate the specified model + dataset
    predict     Use a trained model to make predictions.
    serve       Run the web service and demo.
    make-vocab  Create a vocabulary
    elmo        Use a trained model to make predictions.
    fine-tune   Continue training a model on a new dataset
    dry-run     Create a vocabulary, compute dataset statistics and other
                training utilities.
    test-install
                Run the unit tests.
```

It's what we'll be using throughout this tutorial.

Eventually you'll want to create your own models and helper classes,
at which point you'll need to create your own run script that knows
about them.

### Serving the Demo

The `serve` command starts the demo server.
The first time you run it, it will download
several large serialized models from Amazon S3.

```
$ allennlp serve
2018-05-22 09:36:07,565 - INFO - allennlp.service.server_flask - Starting a flask server on port 8000.
2018-05-22 09:36:07,568 - INFO - allennlp.service.db - Relevant environment variables not found, so no demo database
```

(Currently `serve` doesn't work if you installed using `pip`,
 as the static files for the demo website don't get installed. We're working on it.)

If you now visit `http://localhost:8000` in your browser, you can play around with the same demo
that runs on the AllenNLP website.

![Screenshot of demo](demo.png)

### Next Steps

Continue on to the [Training and Evaluating Models](training_and_evaluating.md) tutorial.
