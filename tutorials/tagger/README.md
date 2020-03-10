# Getting Started with AllenNLP

* [Getting Started](#getting-started)
* [The Problem](#the-problem)
* [The Data](#the-data)
* [Our Task](#our-task)
* [The `DatasetReader`](#the-datasetreader)
* [Defining a `Model`](#defining-a-model)
* [Training the Model](#training-the-model)
* [Making Predictions](#making-predictions)
* [Using Config Files](#using-config-files)
* [Solved Exercise: Adding Character-Level Features](#solved-exercise-adding-character-level-features)
* [Bonus: Creating a Simple Demo](#bonus-creating-a-simple-demo)

## Getting Started

Welcome to AllenNLP! This tutorial will walk you through the basics of building and training an AllenNLP model.
Before we get started, make sure you have a clean Python 3.6 or 3.7 virtual environment, and then run

```
pip install allennlp
```

to install the AllenNLP library.

In this tutorial we'll implement a slightly enhanced version of the PyTorch
[LSTM for Part-of-Speech Tagging](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging) tutorial,
adding some features that make it a slightly more realistic task (and that also showcase some of the benefits of AllenNLP).

## The Problem

Given a sentence (e.g. "The dog ate the apple") we want to predict part-of-speech tags for each word (e.g ["DET", "NN", "V", "DET", "NN"]).

As in the PyTorch tutorial, we'll embed each word in a low-dimensional space, pass them through an LSTM to get a sequence of encodings, and use a feedforward layer to transform those into a sequence of logits (corresponding to the possible part-of-speech tags).

The PyTorch tutorial example is extremely simple, so we'll add a few small twists to make it slightly more realistic:

1. We'll read our data from files. (The PyTorch example uses data that's given as part of the Python code.)
2. We'll use a separate validation dataset to check our performance. (The PyTorch example trains and evaluates on the same dataset.)
3. We'll use [`tqdm`](https://github.com/tqdm/tqdm) to track the progress of our training.
4. We'll implement [early stopping](https://en.wikipedia.org/wiki/Early_stopping) based on the loss on the validation dataset.
5. We'll track accuracy on both the training and validation sets as we train the model.

We won't go through it in detail, but the vanilla PyTorch code (with these modifications)
can be found in [basic_pytorch.py](/tutorials/tagger/basic_pytorch.py).

## The Data

The data can be found in [training.txt](/tutorials/tagger/training.txt) and [validation.txt](/tutorials/tagger/validation.txt). It has one sentence per line, formatted as follows:

```
The###DET dog###NN ate###V the###DET apple###NN
Everybody###NN read###V that###DET book###NN
```

Each sentence consists of space-separated word-tag pairs, which are themselves separated by `###`.

## Our Task

Typically to solve a problem like this using AllenNLP, you'll have to implement two classes.

The first is a [`DatasetReader`], which contains the logic for reading a file of data
and producing a stream of `Instance`s (more about those shortly).

The second is a [`Model`](https://docs.allennlp.org/master/api/models/model/#model), which is a PyTorch `Module` that takes `Tensor` inputs and produces a dict of `Tensor` outputs (including the training `loss` you want to optimize).

AllenNLP handles the remaining details such as training, batching, padding, logging, model persistence, and so on.
AllenNLP also includes many high-level abstractions that will make writing those two classes much easier.

## The `DatasetReader`

### What is a `DatasetReader`?

In AllenNLP each training example is represented as an `Instance` consisting of `Field`s of various types.
A `DatasetReader` contains the logic to generate those instances (typically) from data stored on disk.

Typically to create a `DatasetReader` you'd implement two methods:

1. `text_to_instance` takes the inputs corresponding to a training example
   (in this case the `tokens` of the sentence and the corresponding part-of-speech `tags`),
   instantiates the corresponding `Field`s
   (in this case a `TextField` for the sentence and a `SequenceLabelField` for its tags),
   and returns the `Instance` containing those fields.

2. `_read` takes the path to an input file and returns an `Iterator` of `Instance`s.
   (It will probably delegate most of its work to `text_to_instance`.)

### Constructing a `DatasetReader`

Usually a `DatasetReader` will need to have a `dict` of `TokenIndexer`s
that specify how you want to convert text tokens into indices. For instance, you
will usually have a `SingleIdTokenIndexer` which maps each word to a unique ID,
and you might also (or instead) have a `TokenCharactersIndexer`, which maps
each word to a sequence of indices corresponding to its characters.

In this case we'll only use word IDs, but as good practice we'll
allow users to pass in an alternate set of token indexers if they so desire:

```python
class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
```

The base `DatasetReader` constructor has a parameter
that allows for "lazy" reading of large datasets,
but our datasets only have two sentences each, so we just specify `False`.

### Implementing `text_to_instance`

Next we need to implement the method that creates `Instance`s:

```python
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)
```

A couple of things to notice. The first is that the `tokens` variable is a `List[Token]` (and not a `List[str]`).
If you use the `spacy` tokenizer (which is what our default `SpacyTokenizer` does), that's already the output you get.
If (like us) you have pre-tokenized data, you just need to wrap each string token in a call to `Token`.

Another thing to notice is that the `tags` are optional.
This is so that after we train a model we can use it to make predictions
on untagged data (which clearly won't have any tags).

We define a `TextField` to hold the sentence (you can see that it needs to know the token indexers from the constructor),
and if tags are provided we put them in a `SequenceLabelField`, which is for labels corresponding to each
element of a sequence. (If we had a label that applied to the entire sentence, for example "sentiment", we
would instead use a `LabelField`.)

Finally, we just return an `Instance` containing the dict `field_name` -> `Field`.

### Implementing `_read`

The last piece to implement is `_read`, which takes a filename and produces a stream of `Instance`s.
Luckily, most of the work has already been done in `text_to_instance`:

```python
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)
```

We split each line on spaces to get pairs `word###TAG`,
split each pair to get tuples `(word, tag)`,
use `zip` to break those into a list of words and a list of tags,
wrap each word in `Token` (as described above),
and then call `text_to_instance`.

(The reason for splitting the logic into two functions is that `text_to_instance`
 is useful on its own, for instance, if you build an interactive demo for your model and want to produce
 `Instance`s from user-supplied sentences.)

 And that's it for the dataset reader.

 ## Defining a `Model`

 The other piece you'll need to implement when solving a new problem using `AllenNLP` is a `Model`.
 A `Model` is a subclass of `torch.nn.Module` with a `forward` method that takes some input tensors
 and produces a dict of output tensors. How this all works is largely up to you -- the only requirement
 is that your output dict contain a `"loss"` tensor, as that's what our training code uses to optimize your
 model parameters.

### Constructing the Model

 As in the PyTorch tutorial we're copying, our model will consist of an embedding layer,
 a sequence encoder, and a feedforward network. One thing we'll do that might seem unusual
 is that we're going to _inject_ two of those into our model:

```python
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
```

The embedding layer is specified as an AllenNLP `TextFieldEmbedder`,
which represents a general way of turning tokens into tensors.
(Here we know that we want to represent each unique word with a learned tensor,
 but using the general class allows us to easily experiment with different
 types of embeddings.)

Similarly, the `encoder` is specified as a general `Seq2SeqEncoder`
even though we know we want to use an LSTM. Again, this makes it easy
to experiment with other sequence encoders.

Every AllenNLP `Model` also expects a `Vocabulary`, which contains the namespaced
mappings of tokens to indices and labels to indices. You can see that we have to pass
it to the base class constructor.

The feed forward layer is not passed in as a parameter, but is constructed by us.
Notice that it looks at the encoder to find the correct input dimension
and looks at the vocabulary (and, in particular, at the label -> index mapping)
to find the correct output dimension.

The last thing to notice is that we also instantiate a `CategoricalAccuracy` metric,
which we'll use to track accuracy during each training and validation epoch.

Because of the dependency injection, that's all we have to do to construct the model.

### Implementing `forward`

Next we need to implement `forward`, which is where the actual computation happens.
Each `Instance` in your dataset will get (batched with other instances and) fed into
`forward`. As mentioned above, `forward` expects dicts of tensors as input,
and it expects their names to be the names of the fields in your `Instance`.

In this case we have a `sentence` field and (possibly) a `labels` field,
so we'll construct our `forward` accordingly:

```python
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
```

AllenNLP is designed to operate on batched inputs, but different input sequences have
different lengths. Behind the scenes AllenNLP is padding the shorter inputs so that the batch
has uniform shape, which means our computations need to use a `mask` to exclude the padding.

So we start by passing the `sentence` tensor (each sentence a sequence of token ids)
to the `word_embeddings` module, which converts each sentence into a sequence of embedded tensors.
We then call the utility function `get_text_field_mask`, which returns a tensor of 0s and 1s
corresponding to the padded and unpadded locations.

We next pass the embedded tensors (and the mask) to the LSTM, which produces a sequence of encoded outputs.
Finally, we pass each encoded output tensor to the feedforward layer to produce logits
corresponding to the various tags.

As before, the `labels` were optional, as we might want to run this model to make predictions on unlabeled data.
If we do have labels, then we use them to update our accuracy metric and compute the `"loss"` that goes in our output.
That's all.

### Getting Metrics

We included an accuracy metric that gets updated each forward pass.
That means we need to override a `get_metrics` method that pulls the data out of it:

```python
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
```

Behind the scenes, the `CategoricalAccuracy` metric is storing the number
of predictions and the number of correct predictions, updating those counts
during each call to `forward`. Each call to `get_metric` returns the calculated
accuracy and (optionally) resets the counts, which is what allows us to track accuracy
anew for each epoch.

## Training the Model

Now that we've implemented the `DatasetReader` and the `Model`, we're ready to train.

### Reading the Data

We'll start off by reading the data. Here we read them from URLs, but
you could also read them from local files if you had the data locally.
The `cached_path` helper downloads the files at the URLs to a local cache
and returns the local path to them.

```python
reader = PosDatasetReader()
train_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/training.txt'))
validation_dataset = reader.read(cached_path(
    'https://raw.githubusercontent.com/allenai/allennlp'
    '/master/tutorials/tagger/validation.txt'))
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
```

We instantiate the reader, read both the train and validation datasets
(each of which is a list of `Instance`s),
and use them to create a `Vocabulary` (that is, a mapping from tokens / labels to ids).

### Instantiating the Model

Now we need to construct the model. Remember that we need to pass it a `TextFieldEmbedder`
and a `Seq2SeqEncoder` (as well as a `Vocabulary`, which we already have).

Let's start with the text field embedder. We'll just use the `BasicTextFieldEmbedder`,
which takes a mapping from `index_name` to `Embedding`. If you go back to where we defined
the dataset reader, our default parameters included a single index `"tokens"`.
So our mapping just needs to contain an embedding corresponding to that index:

```python
EMBEDDING_DIM = 6

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
```

Here we use the vocabulary to find the number of embeddings we need,
and the `EMBEDDING_DIM` constant to specify the size of the outputs.

We also need to specify the `Seq2SeqEncoder`:

```python
HIDDEN_DIM = 6

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
```

The need for `PytorchSeq2SeqWrapper` here is slightly unfortunate
(and later we'll show you how not to have to worry about it)
but it's required to add some extra functionality
(and a cleaner interface)
to the built in PyTorch module.
In AllenNLP we do everything batch first, so we specify that as well.

Finally, we can instantiate the model:

```python
model = LstmTagger(word_embeddings, lstm, vocab)
```

### Setting up the `Trainer`

AllenNLP includes a very full-featured `Trainer` class
that handles most of the gory details of training models.
We'll need to pass it our model and our datasets, of course.

We also need to give it an optimizer. Here we'll just use
the PyTorch stochastic gradient descent:

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

And we also need to give a `DataIterator` that
indicates how to batch our datasets:

```python
iterator = BasicIterator(batch_size=2)
iterator.index_with(vocab)
```

We also specify that the iterator should make sure its instances are indexed
using the provided vocabulary.

Finally, we can instantiate our `Trainer` and run it:

```python
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)

trainer.train()
```

Notice that we told it to run for 1000 epochs and that it should stop training early
if it ever spends 10 epochs without the validation loss improving.

When we launch it it will print a progress bar for each epoch that also indicates
the `loss` and `accuracy` metrics. If we've chosen a good model, the `loss` should go down
and the `accuracy` should go up as we train.

```
accuracy: 0.6667, loss: 0.7690 ||: 100%|███████| 1/1 [00:00<00:00, 276.40it/s]
```

At the end of which we'll have a trained model.

## Making Predictions

As in the original PyTorch tutorial, we'd like to look at the predictions our model generates.
AllenNLP contains a `Predictor` abstraction that takes inputs, converts them to `Instance`s,
feeds them through your model, and returns JSON-serializable results.

Often you'd need to implement your own `Predictor`, but AllenNLP already has a `SentenceTaggerPredictor`
that works perfectly here, so we can use it:

```python
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
```

The predictor requires our model (for making predictions)
and our dataset reader (for creating `Instance`s).
It has a `predict` method that just needs a sentence
and that returns (a JSON-serializable version of) the output dict from `Model.forward`:

```python
tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
```

Here `tag_logits` will be a `(5, 3)` array of logits,
where each row represents the scores for one word for the three tag choices.

To get the actual "predictions", we can first find the highest score in each row:

```python
tag_ids = np.argmax(tag_logits, axis=-1)
```

And then use our `Vocabulary` to find the corresponding tags:

```python
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
# ['DET', 'NN', 'V', 'DET', 'NN']
```

## Using Config Files

Although the preceding gives a very detailed walkthrough of how to create and train a model,
in practice you wouldn't do most of that work manually.

### Configuration and Params

Most AllenNLP objects are constructible from JSON-like parameter objects.

So, for instance, instead of writing

```python
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
```

You could write

```python
lstm_params = Params({
    "type": "lstm",
    "input_size": EMBEDDING_DIM,
    "hidden_size": HIDDEN_DIM
})

lstm = Seq2SeqEncoder.from_params(lstm_params)
```

That might not seem like much a win, except for a couple of things:

1. This means that most of your experiment can be specified _declaratively_
   in a separate configuration file, which serves as a record of exactly what
   experiments you ran with which parameters.
2. Now you can change various aspects of your model without writing any code.
   For instance, if you wanted to use a GRU instead of an LSTM, you'd just need
   to change the appropriate entry in the configuration file.

Our configuration files are written in [Jsonnet](https://jsonnet.org/),
which is a superset of JSON with some nice features around variable substitution.

### Using the Config File Approach with Our Model

To use the config file approach, you'll still need to implement the
`PosDatasetReader` and `LstmTagger` classes, the same as before.

The goal is that e.g. `DatasetReader.from_params` should be able to produce a `PosDatasetReader`.
To do this, we need to register it with a type. We provide a decorator that does this:

```python
@DatasetReader.register('pos-tutorial')
class PosDatasetReader:
    ...
```

Once this code has been run, AllenNLP knows that a dataset reader config with type `"pos-tutorial"`
refers to this class. Similarly, we need to decorate our model:

```python
@Model.register('lstm-tagger')
class LstmTagger:
    ...
```

Other than those decorators, the model and dataset reader can remain exactly the same.

But now the remainder of the configuration is specified in [experiment.jsonnet](/tutorials/tagger/experiment.jsonnet).
For the most part it should be pretty straightforward;
one novel piece is that Jsonnet allows us to use local variables,
which means we can specify experimental parameters all in one place.

The config file specifies every detail we did in our by-hand version, which means that
training the model is as simple as

```python
params = Params.from_file('tutorials/tagger/experiment.jsonnet')
serialization_dir = tempfile.mkdtemp()
model = train_model(params, serialization_dir)
```

You can see this version in [config_allennlp.py](/tutorials/tagger/config_allennlp.py).

### Using the Command Line Tool

In fact, we provide a command line tool that handles most common AllenNLP tasks,
so in practice you probably wouldn't read the params or call `train_model` yourself
(although you can), you'd just run the command

```bash
$ allennlp train tutorials/tagger/experiment.jsonnet \
                 -s /tmp/serialization_dir \
                 --include-package tutorials.tagger.config_allennlp
```

The serialization directory is where AllenNLP writes out its serialized model,
its training logs, training checkpoints, and various other things. Mostly you'd
be interested in the serialized model and the logs.

The one that's non-obvious is `--include-package`.
The decorator that registers your classes only runs
when the module containing it is loaded. And the `allennlp` execution
script has no way of knowing it needs to load the modules containing
your custom code (indeed, it doesn't even know those modules exist).
And so the `--include-package` argument tells AllenNLP to load the
specified modules (and in particular run their `register` decorators)
before instantiating and training your module.

And that's it!

## Solved Exercise: Adding Character-Level Features

The PyTorch tutorial [poses a challenge at the end](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#exercise-augmenting-the-lstm-part-of-speech-tagger-with-character-level-features):

    Let’s augment the word embeddings with a representation derived from the characters of the word. We expect that this should help significantly, since character-level information like affixes have a large bearing on part-of-speech. For example, words with the affix -ly are almost always tagged as adverbs in English.

    To do this, let cw be the character-level representation of word w. Let xw be the word embedding as before. Then the input to our sequence model is the concatenation of xw and cw. So if xw has dimension 5, and cw dimension 3, then our LSTM should accept an input of dimension 8.

    To get the character level representation, do an LSTM over the characters of a word, and let cw be the final hidden state of this LSTM.

One of the great benefits of AllenNLP is that it makes this sort of modification extremely simple.

### Changing the Jsonnet Variables

In the original version we used variables in our Jsonnet file:

```
local embedding_dim = 6;
local hidden_dim = 6;
local num_epochs = 1000;
local patience = 10;
local batch_size = 2;
local learning_rate = 0.1;
```

Now we'll have separate embeddings for words and characters,
which means we need to make a change like:

```
local word_embedding_dim = 5;
local char_embedding_dim = 3;
local embedding_dim = word_embedding_dim + char_embedding_dim;
```

If we use those variables correctly, then we only need to change
the word-embedding dimension in a single place to experiment
with different models.

### Adding Character-Level Indexing

The first big change is that we need to add a second `TokenIndexer` to our dataset reader
that captures the character-level indices. As the token indexers can be provided as a
constructor input, we can accomplish this with just a small change to our configuration file:

```jsonnet
    "dataset_reader": {
        "type": "pos-tutorial",
        "token_indexers": {
            "tokens": { "type": "single_id" },
            "token_characters": { "type": "characters" }
        }
    },
```

We now have to explicitly specify the token indexers
(previously we skipped that part and just used the default).
And we have to add a second one that indexes characters.

Where did the name "characters" come from? If you look at the code
 for [`TokenCharactersIndexer`](https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/token_characters_indexer.py#L14), you can see that that's the name it was registered under.

The name `"token_characters"` is our choice -- we'll need to use it as a key
when we specify token embedders inside our model.

### Adding Character-Level Encoding

The section of the configuration corresponding to our model was previously

```jsonnet
    "model": {
        "type": "lstm-tagger",
        "word_embeddings": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
    },
```

The `BasicTokenEmbedder` already knows that if it has multiple token embedders it
should concatenate their outputs. So all we need to do is add a second token embedder
with the right key:

```jsonnet
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": word_embedding_dim
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": char_embedding_dim,
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": char_embedding_dim,
                        "hidden_size": char_embedding_dim
                    }
                }
            },
```

First we change the `embedding_dim` for tokens to `word_embedding_dim`.
Then we add a [character encoding](https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/token_characters_encoder.py#L11), which if you look at its code expects both
an `Embedding` (for turning characters into tensors)
and a `Seq2VecEncoder` (for turning a sequence of tensors into a single tensor).

The embedding is the same type of embedding we used for the word tokens
(but with a different dimension),
and our `Seq2VecEncoder` will just be an LSTM with the same input and hidden dim.

Because of the way we defined the variables in the config file, the
subsequent LSTM automatically has the right dimensions.

### That's All!

Those small changes to the configuration file are all that's needed
to solve the "add character level encodings exercise"!
The enhanced model is now ready to train:

```bash
$ allennlp train tutorials/tagger/exercise.jsonnet \
                 -s /tmp/serialization_dir_exercise \
                 --include-package tutorials.tagger.config_allennlp
```

A couple of things to note:

1. What could have been a fairly involved exercise
   (and should have involved writing code)
   only required a few changes to the Jsonnet configuration file.
   This ease of experimentation is one of the primary benefits of using AllenNLP.

2. That said, *what changes to make* is not obvious for a newcomer to the library
   and requires a reasonable understanding of how AllenNLP works. We hope this
   tutorial has helped you toward such an understanding!

## Bonus: Creating a Simple Demo

Training a model produces a `model.tar.gz` file containing the model architecture,
vocabulary, and weights. In the previous example the file will be located at

```
/tmp/serialization_dir_exercise/model.tar.gz
```

Using a trained model it's easy to run a simple text-in-JSON-out demo. First, install the demo code:

```bash
git clone https://github.com/allenai/allennlp-server.git
pip install allennlp-server/requirements.txt
```

Now we are ready to run our simple demo!
```bash
python allennlp-server/server_simple.py \
    --archive-path /tmp/serialization_dir_exercise/model.tar.gz \
    --predictor sentence-tagger \
    --title "AllenNLP Tutorial" \
    --field-name sentence \
    --include-package tutorials.tagger.config_allennlp \
    --port 8234
```

It requires the path to the trained model archive,
the registered name of the predictor to use,
the field names that the predictor expects,
any extra packages to include, and optionally a title and port.

After a moment you should get a `Model loaded, serving demo on port 8234` message.

If you navigate your browser to `localhost:8234`, you'll get an attractive demo
that runs your model:

![attractive demo](/tutorials/tagger/simple_demo.png)

The tag logits are not the most elegant visualization of what's going on in your model, but they're extremely helpful for debugging, and you got them basically for free!

## Thanks for reading!

Please let us know if you have any feedback on the tutorial,
if any parts are unclear, or if there are things we could add
to make it better!
