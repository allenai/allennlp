# Training a Model

### Training a Model

In this tutorial we'll train a simple part-of-speech tagger using AllenNLP.
The model is defined in [allennlp/models/simple_tagger.py](https://github.com/allenai/allennlp/blob/master/allennlp/models/simple_tagger.py).
It consists of a word embedding layer followed by an LSTM.

Our dataset will be a subset of the [Brown Corpus](http://www.nltk.org/nltk_data/).
In particular, we will train a model on 4000 randomly chosen sentences (`sentences.small.train`) and use a different ~1000 randomly chosen sentences
as the validation set (`sentences.small.dev`).

One of the key design principles behind AllenNLP is that
you configure experiments using JSON files. (More specifically, [HOCON](https://github.com/typesafehub/config/blob/master/HOCON.md) files.)

Our tagging experiment is defined in
[tutorials/getting_started/simple_tagger.json](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/simple_tagger.json).
You can peek at it now if you want; we'll go through it in detail in the next tutorial.
Right at this instant you might care about the `trainer` section, which specifies how we want to train our model:

```js
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1
  }
```

Here the `num_epochs` parameter specifies that we want to make 40 training passes through the training dataset.
On a recent Macbook each epoch of this model on this dataset takes about a minute,
so this training should take about 40, unless it stops early. `patience`
controls the early stopping -- if our validation metric doesn't improve for
this many epochs, training halts. And if you have a GPU you can change `cuda_device` to 0 to use it.

Change any of those if you want to, and then run

```
$ python -m allennlp.run train tutorials/getting_started/simple_tagger.json --serialization-dir /tmp/tutorials/getting_started
```

The `serialization-dir` argument specifies the directory where the model's vocabulary and checkpointed weights will be saved.

This command will download the datasets and cache them locally,
log all of the parameters it's using,
and then display the progress and results of each epoch:

```
2017-08-23 18:07:14,700 - INFO - allennlp.training.trainer - Epoch 2/40
accuracy: 0.51, loss: 2.06, accuracy3: 0.67 ||: 100%|##########| 125/125 [01:08<00:00,  2.03it/s]
accuracy: 0.61, loss: 1.65, accuracy3: 0.75 ||: 100%|##########| 32/32 [00:06<00:00,  4.96it/s]
2017-08-23 18:08:29,397 - INFO - allennlp.training.trainer - Training accuracy : 0.506099    Validation accuracy : 0.606811
2017-08-23 18:08:29,398 - INFO - allennlp.training.trainer - Training loss : 2.061412    Validation loss : 1.646712
2017-08-23 18:08:29,398 - INFO - allennlp.training.trainer - Training accuracy3 : 0.672000    Validation accuracy3 : 0.753761
2017-08-23 18:08:29,423 - INFO - allennlp.training.trainer - Best validation performance so far. Copying weights to /tmp/tutorials/getting_started/best.th'.
```

Here `accuracy` measures how often our model predicted the "correct" part of speech tag as most probable,
while `accuracy3` measures how often the correct tag was one of the _three_ most probable.
`loss` measures [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)
 and is the objective being used to train the model. You want to make sure
 it's mostly decreasing during training.

After 30 epochs the performance on the validation set seems to top out:

```
2017-08-23 18:40:46,632 - INFO - allennlp.training.trainer - Epoch 30/40
accuracy: 0.97, loss: 0.10, accuracy3: 1.00 ||: 100%|##########| 125/125 [01:04<00:00,  1.86it/s]
accuracy: 0.92, loss: 0.40, accuracy3: 0.97 ||: 100%|##########| 32/32 [00:05<00:00,  5.58it/s]
2017-08-23 18:41:57,236 - INFO - allennlp.training.trainer - Training accuracy : 0.966363    Validation accuracy : 0.916993
2017-08-23 18:41:57,236 - INFO - allennlp.training.trainer - Training loss : 0.098178    Validation loss : 0.401380
2017-08-23 18:41:57,237 - INFO - allennlp.training.trainer - Training accuracy3 : 0.995176    Validation accuracy3 : 0.973490
```

This means that 92% of the time our model predicted the correct tag on the validation dataset,
and 97% of the time the correct tag was in the model's "top 3".
Not ground-breaking performance, but this is a pretty simple model, and
if you look at the data there's a lot of different tags!

Now that the model is trained, there should be a bunch of files in the serialization directory. The `vocabulary` directory
contains the model's vocabularies, each of which is a (distinct) encoding of strings as integers.
In our case, we'll have one for `tokens` (i.e. words) and another for `tags`. The various
`training_state_epoch_XX.th` files contain the state of the trainer after each epoch (`.th` is the suffix for serialized torch tensors),
so that you could resume training where you left off, if you wanted to.
Similarly, the `model_state_epoch_XX.th` files contain the model weights after each epoch.
`best.th` contains the *best* weights (that is, those from the epoch with the smallest `loss` on the validation dataset).

Finally, there is an "archive" file `model.tar.gz` that contains the training configuration,
the `best` weights, and the `vocabulary`.

### Evaluating a Model

Once you've trained a model, you likely want to evaluate it on another dataset.
We have another 1000 sentences in the file `sentences.small.test`, which
is shared publicly on Amazon S3.

We can use the `evaluate` command, giving it the archived model and the evaluation dataset:

```
$ python -m allennlp.run evaluate /tmp/tutorials/getting_started/model.tar.gz --evaluation-data-file https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.test
```

When you run this it will load the archived model, download and cache the evaluation dataset, and then make predictions:

```
2017-08-23 19:49:18,451 - INFO - allennlp.models.archival - extracting archive file /tmp/tutorials/getting_started/model.tar.gz to temp dir /var/folders/_n/mdsjzvcs6s705kpn87f399880000gp/T/tmptgu44ulc
2017-08-23 19:49:18,643 - INFO - allennlp.commands.evaluate - Reading evaluation data from https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.test
2017-08-23 19:49:18,643 - INFO - allennlp.common.file_utils - https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.test not found in cache, downloading to /Users/joelg/.allennlp/datasets/aHR0cHM6Ly9hbGxlbm5scC5zMy5hbWF6b25hd3MuY29tL2RhdGFzZXRzL2dldHRpbmctc3RhcnRlZC9zZW50ZW5jZXMuc21hbGwudGVzdA==
100%|████████████████████████████████████████████████████████████████████████████████████| 170391/170391 [00:00<00:00, 1306579.69B/s]
2017-08-23 19:49:20,203 - INFO - allennlp.data.dataset_readers.sequence_tagging - Reading instances from lines in file at: /Users/joelg/.allennlp/datasets/aHR0cHM6Ly9hbGxlbm5scC5zMy5hbWF6b25hd3MuY29tL2RhdGFzZXRzL2dldHRpbmctc3RhcnRlZC9zZW50ZW5jZXMuc21hbGwudGVzdA==
1000it [00:00, 36100.84it/s]
2017-08-23 19:49:20,233 - INFO - allennlp.data.dataset - Indexing dataset
100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 7155.68it/s]
2017-08-23 19:49:20,373 - INFO - allennlp.commands.evaluate - Iterating over dataset
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:05<00:00,  5.47it/s]
2017-08-23 19:49:26,228 - INFO - allennlp.commands.evaluate - Finished evaluating.
2017-08-23 19:49:26,228 - INFO - allennlp.commands.evaluate - Metrics:
2017-08-23 19:49:26,228 - INFO - allennlp.commands.evaluate - accuracy: 0.9070572302753674
2017-08-23 19:49:26,228 - INFO - allennlp.commands.evaluate - accuracy3: 0.9681496714651151
```

There is also a command line option to use a GPU, if you have one.

### Making Predictions

Finally, what's the good of training a model if you can't use it to make predictions?
The `predict` command takes an archived model and a [JSON lines](https://en.wikipedia.org/wiki/JSON_Streaming#Line_delimited_JSON)
file of inputs and makes predictions using the model.

Here, the "predictor" for the tagging model expects a JSON blob containing a sentence:

```bash
$ cat <<EOF >> inputs.txt
{"sentence": "I am reading a tutorial."}
{"sentence": "Natural language processing is easy."}
EOF
```

After which we can make predictions:

```bash
$ python -m allennlp.run predict /tmp/tutorials/getting_started/model.tar.gz inputs.txt
... lots of logging omitted
{"tags": ["ppss", "bem", "vbg", "at", "nn", "."], "class_probabilities": [[ ... ]]}
{"tags": ["jj", "nn", "nn", "bez", "jj", "."], "class_probabilities": [[ ... ]]}
```

Here the `"tags"` are the part-of-speech tags for each sentence, and the
`"class_probabilities"` are the predicted distributions of tags for each sentence
(and are not shown above, as there are a lot of them).

### Next Steps

Continue on to our [Configuration](configuration.md) tutorial.
