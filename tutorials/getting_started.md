# Getting Started

Welcome to AllenNLP!

## Installing using Docker

The easiest way to get started is using Docker. Assuming you have Docker installed, just run

```bash
docker run -p 8000:8000 -it --rm allennlp/allennlp-cpu
```

(If your machine has GPUs, use `allenlp-gpu` instead.)

This will pull down the latest `allennlp` image to your machine
(which may take a minute the first time you do it),
start a Docker container, and launch an interactive shell.
(It also exposes port 8000, which is where the demo server runs,
 and shuts down the container when you exit the interactive shell.)

## Installing Not Using Docker

If you can't (or don't want to) use Docker, you can clone from our git repository:

```bash
git clone https://github.com/allenai/allennlp.git
```

Create a Python 3.5 (or 3.6) virtual environment, and run

```bash
INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
```

changing the flag to `false` if you don't want to be able to run tests.
(Narrator: You want to be able to run tests.)

You'll also need to install PyTorch 0.2, following the appropriate instructions
from [their website](http://pytorch.org/).

## Once You've Installed

A lot of the most basic fuctionality can be accessed through the command line tool `allennlp/run`:

```
(allennlp) root@9175b60b4e52:/stage# allennlp/run
usage: run [command]

Run AllenNLP

optional arguments:
  -h, --help  show this help message and exit

Commands:

    bulk      Run a model in bulk.
    train     Train a model
    serve     Run the web service and demo.
    evaluate  Evaluate the specified model + dataset
```

### Serving the Demo

The `serve` command

```
(allennlp) root@9175b60b4e52:/stage# allennlp/run serve
Starting a flask server on port 8000.
[... lots of logging omitted ...]
2017-08-14 20:00:48,330 - INFO - werkzeug -  * Running on http://0.0.0.0:8000/ (Press CTRL+C to quit)
```

If you navigate your browser to `http://localhost:8000`, you will see the same demo
that runs on the AllenNLP website.

TODO(joelgrus): screenshot

### Training a Model

AllenNLP experiments are defined using JSON files that specify
model parameters, training parameters, data paths, and metrics to compute.

TODO(joelgrus): create a spec / dataset just for this tutorial

The specification for a very stripped-down BiDAF model can be found in `tests/fixtures/bidaf/experiment.json`.


