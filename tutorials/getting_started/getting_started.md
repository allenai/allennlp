---
layout: tutorial
title: Getting Started
id: getting-started
---

Welcome to AllenNLP!

## Installing using Docker

The easiest way to get started is using Docker. Assuming you have Docker installed, just run

```bash
docker run -p 8000:8000 -it --rm allennlp/allennlp-cpu
```

If your machine has GPUs, use `allennlp-gpu` instead.

This will download the latest `allennlp` image to your machine
(unless you already have it),
start a Docker container, and launch an interactive shell.
It also exposes port 8000, which is where the demo server runs,
and shuts down the container when you exit the interactive shell.

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

A lot of the most common functionality can be accessed through the command line tool `allennlp/run`:

```
(allennlp) root@9175b60b4e52:/stage# allennlp/run
usage: run [command]

Run AllenNLP

optional arguments:
  -h, --help  show this help message and exit

Commands:

    predict   Use a trained model to make predictions.
    train     Train a model
    serve     Run the web service and demo.
    evaluate  Evaluate the specified model + dataset
```

### Serving the Demo

The `serve` command starts the demo server.

```
(allennlp) root@9175b60b4e52:/stage# allennlp/run serve
Starting a sanic server on port 8000.
[... lots of logging omitted ...]
2017-08-16 18:55:12 - (sanic)[INFO]: Goin' Fast @ http://0.0.0.0:8000
2017-08-16 18:55:12,321 - INFO - sanic - Goin' Fast @ http://0.0.0.0:8000
2017-08-16 18:55:12 - (sanic)[INFO]: Starting worker [33290]
2017-08-16 18:55:12,323 - INFO - sanic - Starting worker [33290]
```

If you visit `http://localhost:8000` in your browser, you can play around with the same demo
that runs on the AllenNLP website.

TODO(joelgrus): screenshot

### Training a Model

In this tutorial we'll train a simple part-of-speech tagger using AllenNLP.
The model is defined in [allennlp/models/simple_tagger.py](https://github.com/allenai/allennlp/blob/master/allennlp/models/simple_tagger.py).
It consists of a word embedding layer followed by an LSTM.

Our dataset will be a subset of the [Brown Corpus](http://www.nltk.org/nltk_data/).
In particular, the file [tutorials/getting_started/data/cr.train](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/data/cr.train)
is the concatenation of the Brown files `cr01`, ..., `cr08` (that's the "humor" category)
and [tutorials/getting_started/data/cr.dev](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/data/cr.test) is the file `cr09`.

In AllenNLP we configure experiments using JSON files. Our experiment is defined in
[tutorials/getting_started/simple_tagger.json](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/simple_tagger.json). You can peek at it
if you want, we'll go through it in detail in the next tutorial.  Right at this instant
you might care about the `trainer` section, which specifies how we want to train our model:

```js
  "trainer": {
    "num_epochs": 20,
    "serialization_prefix": "/tmp/tutorials/getting_started",
    "cuda_device": -1
  }
```

Here the `num_epochs` parameter specifies that we want to make 20 training passes through the training dataset.
On a recent Macbook each epoch of this model on this dataset takes about 30 seconds, so 20 will take about 10 minutes.
The `serialization_prefix` is the path where the model's vocabulary and checkpointed weights will be saved.
And if you have a GPU you can change `cuda_device` to 0 to use it.

Change any of those if you want to, and then run

```
allennlp/run train tutorials/getting_started/simple_tagger.json
```

It will log all of the parameters it's using and then display the progress and results of each epoch:

```
2017-08-15 11:37:53,030 - INFO - allennlp.training.trainer - Epoch 5/20
accuracy: 0.48, accuracy_top3: 0.63, loss: 2.19 ||: 100%|##########| 477/477 [00:25<00:00, 21.68it/s]
accuracy: 0.50, accuracy_top3: 0.64, loss: 2.27 ||: 100%|##########| 50/50 [00:00<00:00, 53.72it/s]
2017-08-15 11:38:19,609 - INFO - allennlp.training.trainer - Training accuracy : 0.477715    Validation accuracy : 0.501286
2017-08-15 11:38:19,610 - INFO - allennlp.training.trainer - Training accuracy3 : 0.631720    Validation accuracy3 : 0.642796
2017-08-15 11:38:19,610 - INFO - allennlp.training.trainer - Training loss : 2.187194    Validation loss : 2.265297
2017-08-15 11:38:19,617 - INFO - allennlp.training.trainer - Best validation performance so far. Copying weights to /tmp/tutorials/getting_started/best.th'.
```

Here `accuracy` measures how often our model predicted the "correct" part of speech tag as most probable,
while `accuracy3` measures how often the correct tag was one of the _three_ most probable.
`loss` measures [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
 and is the objective being used to train the model. You want to make sure
 it's mostly decreasing during training.

After 20 epochs we see

```
2017-08-15 13:22:01,591 - INFO - allennlp.training.trainer - Epoch 20/20
accuracy: 0.85, loss: 0.58, accuracy3: 0.92 ||: 100%|##########| 477/477 [00:25<00:00, 19.75it/s]
accuracy: 0.71, loss: 1.38, accuracy3: 0.83 ||: 100%|##########| 50/50 [00:01<00:00, 47.64it/s]
2017-08-15 13:22:28,214 - INFO - allennlp.training.trainer - Training accuracy : 0.849197    Validation accuracy : 0.714408
2017-08-15 13:22:28,214 - INFO - allennlp.training.trainer - Training loss : 0.577224    Validation loss : 1.378879
2017-08-15 13:22:28,214 - INFO - allennlp.training.trainer - Training accuracy3 : 0.924702    Validation accuracy3 : 0.831475
2017-08-15 13:22:28,222 - INFO - allennlp.training.trainer - Best validation performance so far. Copying weights to /tmp/tutorials/getting_started/best.th'.
```

This means that 71% of the time our model predicted the correct tag on the validation dataset,
and 83% of the time the correct tag was in the model's "top 3".
Not ground-breaking performance, but this is a pretty simple model, and
if you look at the data there's a lot of different tags!

Now that the model is trained, there should be a bunch of files in the serialization directory. The `vocabulary` directory
contains the model's vocabularies, each of which is a (distinct) encoding of strings as integers.
In our case, we'll have one for `tokens` (i.e. words) and another for `tags`. The various
`training_state_epoch_XX.th` files contain the state of the trainer after each epoch (`.th` is the suffix for serialized torch tensors),
so that you could resume training where you left off, if you wanted to.
Similarly, the `model_state_epoch_XX.th` files contain the model weights after each epoch.
Finally `best.th` contains the *best* weights (that is, those from the epoch with the smallest `loss` on the validation dataset).

### Evaluating a Model

Once you've trained a model, you likely want to evaluate it on another dataset.
In [tutorials/getting_started/data/cp.test](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/data/cp.test)
we have the contents of Brown Corpus `cp01` (the first file in the "Romance" section).  As our configuration file specifies where
the model was serialized to, it also tells us enough to evaluate the model:

```
allennlp/run evaluate --config_file tutorials/getting_started/simple_tagger.json --evaluation_data_file tutorials/getting_started/data/cp.test
```

This will evaluate the trained model on the evaluation data file:

```
2017-08-15 13:11:47,106 - INFO - allennlp.commands.evaluate - Iterating over dataset
66it [00:00, 67.47it/s]
2017-08-15 13:11:48,084 - INFO - allennlp.commands.evaluate - Finished evaluating.
2017-08-15 13:11:48,084 - INFO - allennlp.commands.evaluate - Metrics:
2017-08-15 13:11:48,084 - INFO - allennlp.commands.evaluate - accuracy: 0.7590051457975986
2017-08-15 13:11:48,084 - INFO - allennlp.commands.evaluate - accuracy3: 0.8421955403087479
```

By default, `evaluate` uses the `best.th` weights. There are command line options to specify non-default weights
and to use a GPU.

### Next Steps

Continue on to our Deep Dive tutorial.
