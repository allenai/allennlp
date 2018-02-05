# Using AllenNLP in your Project

In this tutorial, we take you step-by-step through the process you need to go through to get up and
running with your own models on your own data in your own repository, using AllenNLP as an imported
library.  There is naturally some overlap with things we covered in previous tutorials, but, hey,
some repetition is good, right?

We'll build a model for classifying academic papers, and we'll do it using a [separate
repository](https://github.com/allenai/allennlp-as-a-library-example), showing how to include
AllenNLP as a dependency in your project, what classes you need to implement, and how to train your
model.  We'll go through the code we wrote in that repository, explaining it as we go.

In the end, there are only about 110 lines of actual implementation code in the `DatasetReader`
and `Model` classes that we implement.  This tutorial is long because we're explaining what goes
into those 110 lines, and what AllenNLP does for you behind the scenes so that you only need 110
lines of code to get a very flexible classifier for academic papers.

## First step: install AllenNLP

The first thing we need to do is specify AllenNLP as a dependency in our project.  We'll do this by
adding a
[`requirements.txt`](https://github.com/allenai/allennlp-as-a-library-example/blob/master/requirements.txt)
file.  It contains a single line: `allennlp==0.3`.  Then, after creating a python 3.6 environment,
you install AllenNLP by running `pip install -r requirements.txt`.  You also need to install
pytorch and a spacy model, as described in [the installation tutorial](installation.md):

```bash
pip install -r requirements.txt
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
python -m spacy download en_core_web_sm
```

(The above is assuming CUDA 8 installed on a linux machine; use a different pytorch version as
necessary.)

## Step two: organize your modules

As explained in the [tutorial on creating models](creating_a_model.md), there are two pieces of
code that you have to write when you're building a model in AllenNLP: a `DatasetReader` and a
`Model`.  So the next thing we do is [set up our library to hold this
code](https://github.com/allenai/allennlp-as-a-library-example/tree/master/my_library).  We called
the base module `my_library` - you'll probably want to choose a different name - and included two
submodules: `dataset_readers` and `models`.  We'll put our two classes in these two modules.  You
can use whatever structure you want here; we chose this because it makes it clear what you have to
write, and if you end up writing several different models or working on several tasks, this
structure allows the repository to grow naturally.

## Step three: write your DatasetReader

Now that our repository is set up, we'll start actually writing code.  The first thing we need is
some data - what are we trying to build a model to predict?  In this example, we'll try to predict
the "venue" of academic papers.  That is, given the title and abstract of a paper, we'll try to
decide if it was (or should be) published in a "natural language processing" venue, a "machine
learning" venue, or an "artificial intelligence" venue (all of those scare quotes are because this
is a totally artificial task, and there aren't solid lines between these fields).

We'll use the [open research corpus](http://labs.semanticscholar.org/corpus/) provided by the academic
search engine [Semantic Scholar](http://semanticscholar.org), with a heuristically-edited "venue"
field.  You can follow the link to see the full specification of this data, but it's provided as a
JSON-lines file, where each JSON blob has at least these fields:

```json
{
  "title": "A review of Web searching studies and a framework for future research",
  "paperAbstract": "Research on Web searching is at an incipient stage. ...",
  "venue": "{AI|ML|ACL}"
}
```

Because we like writing tests (and you should too!), we'll write a test for our `DatasetReader` on
some sample data before even writing any code.  We downloaded the data and took a sample of 10
papers and made a little [test
fixture](https://github.com/allenai/allennlp-as-a-library-example/blob/master/tests/fixtures/s2_papers.jsonl)
out of them.  We'll use this fixture to make sure we can read the data as we expect.

For our test, we inherit from `allennlp.common.testing.AllenNlpTestCase`.  For this simple test,
that's not really necessary, but that base class does have some functionality around logging and
cleaning up temporary files that can be nice for more complex tests.

The basic API of the `DatasetReader` is just that we need to instantiate the object, then call
`reader.read(data_file)`.  We do that in our test:

```python
from allennlp.common.testing import AllenNlpTestCase
from my_library.dataset_readers import SemanticScholarDatasetReader

class TestSemanticScholarDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = SemanticScholarDatasetReader()
        dataset = reader.read('tests/fixtures/s2_papers.jsonl')
```

Then we just want to make sure that the resulting dataset looks like we expect.  We'll refer you to
the [dataset tutorial](../notebooks/data_pipeline.ipynb) for a deeper dive on the `Dataset`,
`Instance`, and `Field` classes; for now, just remember that we want each paper to have a title, an
abstract, and a venue. The paper itself is an `Instance` inside of the `Dataset`, and the title, abstract
and venue are all `Fields` inside the `Instance`.  We can make sure that the dataset got read correctly
by giving expected values for the first few instances in our test fixture:

```python
        instance1 = {"title": ["Interferring", "Discourse", "Relations", "in", "Context"],
                     "abstract": ["We", "investigate", "various", "contextual", "effects"],
                     "venue": "ACL"}

        instance2 = {"title": ["GRASPER", ":", "A", "Permissive", "Planning", "Robot"],
                     "abstract": ["Execut", "ion", "of", "classical", "plans"],
                     "venue": "AI"}
        instance3 = {"title": ["Route", "Planning", "under", "Uncertainty", ":", "The", "Canadian",
                               "Traveller", "Problem"],
                     "abstract": ["The", "Canadian", "Traveller", "problem", "is"],
                     "venue": "AI"}
```

The `DatasetReader` needs to tokenize the text that it sees in the titles and abstracts, so those
are given as `Lists` here, but other than that, this just mimics the relevant fields from the JSON
blobs in the data file.  We'll have our test make sure that what we read matches this:

```python
        assert len(dataset.instances) == 10
        fields = dataset.instances[0].fields
        assert [t.text for t in fields["title"].tokens] == instance1["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance1["abstract"]
        assert fields["label"].label == instance1["venue"]
        fields = dataset.instances[1].fields
        assert [t.text for t in fields["title"].tokens] == instance2["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance2["abstract"]
        assert fields["label"].label == instance2["venue"]
        fields = dataset.instances[2].fields
        assert [t.text for t in fields["title"].tokens] == instance3["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance3["abstract"]
        assert fields["label"].label == instance3["venue"]
```

An astute reader might note that we've changed the name of the venue to "label" in the `Instance`
fields - this is because we're thinking ahead to other kinds of labeling tasks we might want to do
with the same data.  Giving the venue the name "label" will let the model that we write later be a
little more general.

With our test in hand, we're now ready to actually write the `DatasetReader` itself.  We'll put it
as a [file in
`my_library.dataset_readers`](https://github.com/allenai/allennlp-as-a-library-example/blob/master/my_library/dataset_readers/semantic_scholar_papers.py).

To understand what's going on in this class, let's look at the `_read` method first:

```python
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(Tqdm.tqdm(data_file.readlines())):
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                title = paper_json['title']
                abstract = paper_json['paperAbstract']
                venue = paper_json['venue']
                yield self.text_to_instance(title, abstract, venue)
```

The `cached_path` method inside of the `with open()` call allows the path we specify to be a web
URL instead of a path on disk; this will be important later so you can use this `DatasetReader`
with some example data we provide, but most of the time this isn't necessary for your code.

The logic inside of `read` just goes through every line in the input file, reading it as a JSON
object, pulling out the fields we want, and passing them to `self.text_to_instance`.
`text_to_instance` takes untokenized text strings and does whatever is necessary to create an
`Instance` for this data.  When you see how simple this method is, you might think it's unnecessary
to have this bit of redirection, instead of just putting that logic inside of `read()` directly.
The reason we have `text_to_instance` is to make it easy to hook up a demo to your model - the demo
needs to process data in the same way that the model's training data was processed, and this
redirection makes that possible.  Here's `text_to_instance`:

```python
    def text_to_instance(self, title: str, abstract: str, venue: str = None) -> Instance:
        tokenized_title = self._tokenizer.tokenize(title)
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        title_field = TextField(tokenized_title, self._token_indexers)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)
        fields = {'title': title_field, 'abstract': abstract_field}
        if venue is not None:
            fields['label'] = LabelField(venue)
        return Instance(fields)
```

The first important thing to notice here is that an `Instance` is made up of a named collection of
`Fields`.  These `Fields` get converted by the rest of the AllenNLP library into batched arrays
that get passed to your `Model`.  A `TextField` represents a tokenized piece of text, a
`LabelField` represents a categorical label, and there are other `Fields` representing other kinds
of data that we don't need to use here.  These inputs are plain strings, but they will be converted
to integer arrays before being passed to your `Model` after building a vocabulary from the training
data.  All of this is handled for you by AllenNLP, in a way that you can configure.

The second important thing to notice is that we're using some objects on `self` to configure how
the `TextFields` are constructed: `self._tokenizer` and `self._token_indexers`.  The `Tokenizer`
determines how the title and abstract get converted into tokens.  The `Tokenizer` could split the
string into words, characters, byte pairs, or any other way you want to tokenize your data.  The
`TokenIndexers` then determine how to represent these tokens as arrays in your model.  If your
tokens are words, for instance, the `TokenIndexers` could make arrays out of the word id, the
characters in the word, the word's part of speech, or many other things.  Notice that we're not
specifying _any_ of that in the code, we're just saying that we'll get a `Tokenizer` and some
`TokenIndexers` as input to this `DatasetReader`, and we'll use them to process our data.  This
lets us be _really_ flexible later in how exactly our inputs get represented.  We can experiment
with using character-level CNNs or POS-tag embeddings without changing our `DatasetReader` or our
`Model` code at all.

How does this work?  This brings us to the last pieces of our `DatasetReader`: the constructor and
the `from_params` method.

```python
@DatasetReader.register("s2_papers")
class SemanticScholarDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @classmethod
    def from_params(cls, params: Params) -> 'SemanticScholarDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers)
```

The constructor takes these dependencies as inputs, where by default we're splitting text into
words and representing them as single word ids (as opposed to, say, sequences of character ids).
To get different tokenization behavior or different word representations, you just need to pass in
different objects when you construct the `DatasetReader`.

Typically, though, you would not be constructing the `SemanticScholarDatasetReader` yourself.
Instead, you specify a [JSON configuration file](configuration.md), and AllenNLP uses that
configuration file to construct your `DatasetReader` and your `Model` for you (among other things).
In order for the library to be able to construct your objects from the JSON in your configuration,
you need to do two things.  First, you need to _register_ your objects with our library, so that
our code can find your class when it tries to instantiate a `DatasetReader`.  That is what the
first line in the code above is doing; we register `SemanticScholarDatasetReader` as a
`DatasetReader` with the name `s2_papers`, which will let us use that name in our JSON
configuration file.  Then, when AllenNLP is trying to construct a `DatasetReader`, it will take the
parameters you specify and pass them to the `from_params` method.  This method takes a `Params`
object, which is just a JSON dictionary with some added functionality, and constructs the
`SemanticScholarDatasetReader`.  All of the `DatasetReader`'s dependencies that we want to be able
to configure from the JSON file need to be constructed here.  In this case, we just create a
`Tokenizer` and a `TokenIndexer` dictionary, using methods that are built in to the library.

And that's it!  In just a few lines of code, we have a flexible `DatasetReader` that will get us
data for our `Model`.  We can run the test we wrote with `pytest` and see that it passes.

## Step four: write your model

With code that will process data for us, we're now ready to build a model for this data.
Once again, we'll start with [a
test](https://github.com/allenai/allennlp-as-a-library-example/blob/master/tests/models/academic_paper_classifier_test.py):

```python
from allennlp.common.testing import ModelTestCase

class AcademicPaperClassifierTest(ModelTestCase):
    def setUp(self):
        super(AcademicPaperClassifierTest, self).setUp()
        self.set_up_model('tests/fixtures/academic_paper_classifier.json',
                          'tests/fixtures/s2_papers.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
```

This time we'll use `allennlp.common.testing.ModelTestCase`.  For any model that you write, you
want to be sure that it can train successfully, that you can save it and load it and get the same
predictions back, and that its predictions are consistent whether the data is batched or not.
`ModelTestCase` has easy tests for all of these.  With just the code above, you can make sure your
model works correctly on a tiny dataset.  We strongly recommend that you write and debug your code
using these tests, as it is way easier and faster to find problems using a test fixture than using
your large dataset.

In order to make use of these tests, you need to provide two things: a [JSON configuration
file](configuration.md) like what you use for training, just with smaller parameters, and a tiny
dataset.  We've already seen the dataset, and we'll examine the configuration file that we're using
here later; it'll make more sense to look at the model code first.

But before we write any model code, let's decide on the basic structure of the model.  We have two
inputs (a title and an abstract) and an output label.  What if we just embed the words in the title
and the abstract, pass them each through a function that converts the sequence of embeddings into a
single vector, and then run a simple feed-forward network on those two vectors to get a label?
Seems like a reasonable first thing to try.

That model structure means we are going to have a few things that we'll want to be able to
configure later - how exactly do I embed the words in the title and the abstract?  How do I combine
the vector sequences into a single vector?  How deep and wide should my feed-forward network be?
Luckily, AllenNLP provides abstractions that encapsulate exactly these operations (and a few more
that we don't need here), which lets you write your model code using the abstract objects, and
configure them with concrete instantiations later.  To start with, then, we need to take these
objects in our constructor:

```python
@Model.register("paper_classifier")
class AcademicPaperClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 abstract_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AcademicPaperClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'AcademicPaperClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        title_encoder = Seq2VecEncoder.from_params(params.pop("title_encoder"))
        abstract_encoder = Seq2VecEncoder.from_params(params.pop("abstract_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   title_encoder=title_encoder,
                   abstract_encoder=abstract_encoder,
                   classifier_feedforward=classifier_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
```

Just as with the `DatasetReader`, we `register` our `Model` and provide a `from_params` method, so
that we can build this `Model` from a JSON configuration file.  Notice, though, that this time
`from_params` additionally takes a `Vocabulary` that's not part of the `Params` object.  The
`Vocabulary` maps strings in your data to integers that can be used by your model code.  AllenNLP
constructs this vocabulary from the data that you use to train your model (exactly how this is
done is configurable, and not covered in this tutorial), then passes it to the `Model`.

Because there are often several different mappings you want to have in your model, the `Vocabulary`
keeps track of separate namespaces.  In this case, we have a "tokens" vocabulary for the text in
the title and the abstract (that's not shown in the code, but is the default value used behind the
scenes), and a "labels" vocabulary for the labels that we're trying to predict.  The
`TextFieldEmbedder`, which creates an embedded representation of a `TextField`, uses the size of
the "tokens" vocabulary to know how big to make its embedding matrices, and the `Model` class
itself uses the size of the "labels" vocabulary to know how many outputs there are.

For converting vector sequences into single vectors, we use a `Seq2VecEncoder`.  This is an
abstraction over pytorch `Modules` that map tensors of shape `(batch_size, sequence_length,
embedding_dim)` to tensors of shape `(batch_size, encoding_dim)`.  The simplest example of a
`Seq2VecEncoder` is a simple bag of words model that averages all of the vectors in the sequence.
RNNs and CNNs can also be used to perform this operation.  AllenNLP has concrete `Seq2VecEncoder`
implementations for all of these options.

`FeedForward` is a simple configurable `Module` that allows specifying the width, depth, and inner
activations of the network.  The `InitializerApplicator` contains a mapping from parameter names to
initialization methods, if you want to use non-default initialization for any of your model's
parameters, and the `RegularizerApplicator` contains a similar mapping for parameter
regularization.

We also define our model's metrics and loss function in the constructor.  `CategoricalAccuracy` is
a `Metric` that calculates and accumulates the model's accuracy in predicting a paper's label.  We
show a running value for these metrics after each batch during training and validation, reseting
the values every epoch.

Now that we understand all of the components that go into this model, the computation performed by
the model is pretty straightforward.  In pytorch, the method that we need to implement for our
`Model` (which is just a pytorch `Module`) is `forward`:

```python
    def forward(self,
                title: Dict[str, torch.LongTensor],
                abstract: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_title = self.text_field_embedder(title)
        title_mask = util.get_text_field_mask(title)
        encoded_title = self.title_encoder(embedded_title, title_mask)

        embedded_abstract = self.text_field_embedder(abstract)
        abstract_mask = util.get_text_field_mask(abstract)
        encoded_abstract = self.abstract_encoder(embedded_abstract, abstract_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_title, encoded_abstract], dim=-1))
        class_probabilities = F.softmax(logits)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict
```

The first thing to notice are the inputs to the method.  Remember the `DatasetReader` we
implemented?  It created `Instances` with fields named `title`, `abstract`, and `label`.  That's
where the names to `forward` come from - they have to match the names that we gave to the fields in
our `DatasetReader`.  AllenNLP will take the instances read by your `DatasetReader`, group them
into batches, pad all of the `Fields` from each instance in the batch to be the same shape, and
then produce one batched array (or set of arrays) for each `Field` in your `Instances`.

Note that we require you to pass the _label_ to `forward`, in addition to the model's inputs - in
order to have a flexible yet sane training loop, we need `Model.forward` to compute its own loss.
The training code will look for the `loss` value in the dictionary returned by `forward`, and
compute the gradients of that loss to update the model's parameters.  But also notice that the
`label` input is _optional_.  This is necessary for you to be able to use this model in situations
where you don't have a label, such as in a demo, or if you want this model to be a component in
some larger model.

Next, let's look at the types of the inputs.  The label is simple: it's just a tensor of shape
`(batch_size, 1)`, with one label id for each instance in the batch.  The other two are a little
more complicated.  Remember that the title and abstract were `TextFields`; those get converted into
a dictionary of pytorch tensors.  It's a dictionary instead of a single tensor to be flexible about
how exactly words are represented in your model.  One element of this dictionary might have a
tensor of word ids, one might have a tensor of character ids for each word, and one might have a
tensor of part of speech tag ids.  But your model doesn't have to care about what exactly is in
that dictionary, because it just passes the dictionary on to the `TextFieldEmbedder` that we took
as a constructor parameter.  You need to be sure that the `TextFieldEmbedder` is expecting the same
thing that your `DatasetReader` is producing, but that happens in the configuration file, and we'll
talk about it later.

Now that we understand the inputs to `forward`, let's look at its logic.  The first thing that the
model does is embed the title and the abstract, then encode them as single vectors.  In order to
encode the title and abstract, we need to get masks representing which elements of the token
sequences are merely there for padding.

Once we have a (batched) single vector for the title and the abstract, we concatenate the two
vectors and pass them through a feed-forward network to get class logits.  We pass the logits
through a softmax to get prediction probabilities.  Lastly, if we were given a label, we can
compute a loss and evaluate our metrics.

There are only two other small pieces of code in the `Model` class.  The first is the
`get_metrics` method:

```python
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
```

This method is how the model tells the training code which metrics it's computing.  The second
pieces of code is the `decode` method:

```python
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict
```

`decode` has two functions: it takes the output of `forward` and does any necessary inference or
decoding on it, and it converts integers into strings to make things human-readable (e.g., for the
demo).  This is a simple model, so we only need to worry about the second function here.

Alright, we've got ourselves a model.  We can run the tests again using `pytest` and confirm that
our model can indeed be trained on our tiny dataset, and that saving and loading works correctly.
Now let's train it on some real data!

## Step five: train the model

As per our [getting started tutorial](training_and_evaluating.md),
you can use `python -m allennlp.run train` to train a model.

To do that, we need a configuration file.  You can see the full file
[here](https://github.com/allenai/allennlp-as-a-library-example/blob/master/experiments/venue_classifier.json).
We'll look at it in chunks.

```json
  "dataset_reader": {
    "type": "s2_papers"
  },
```

This portion contains the parameters for the `DatasetReader`.  The "type" key specifies that we
want to use our `SemanticScholarDatasetReader` (which we registered with the name `s2_papers`),
and the remaining parameters get passed to `SemanticScholarDatasetReader.from_params`.  There
aren't any, so we're just using the default tokenizer and word indexers (which, if you recall,
were to split strings into words and represent words as single ids under the name "tokens").

```json
  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/train.jsonl",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/dev.jsonl",
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["abstract", "num_tokens"], ["title", "num_tokens"]],
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
```

These parameters specify what data to use for training and validation, how to group the data into
batches (that's the "iterator" key), and how to train the model.  The data iterator that we're
using sorts the instances by the number of tokens in the abstract, then the number of tokens in the
title (with a little bit of noise added), then groups them into batches of size 64.  The sorting
helps with efficiency, so you don't waste too much computation on padding tokens.  The trainer uses
the AdaGrad optimizer for 40 epochs, stopping if validation accuracy has not increased for the last
10 epochs.  The data files themselves are given as links to some sample data we've provided on S3;
this is why we needed the `cached_path` call mentioned above.

The last piece of the configuration to look at is the model itself:

```json
  "model": {
    "type": "paper_classifier",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "embedding_dim": 100,
        "trainable": false
      }
    },
    "title_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
   },
```

Here again the "type" key matches what we registered our `Model` as, and the rest of the parameters
get passed to our `AcademicPaperClassifier.from_params` method.  We have parameters for
constructing the `TextFieldEmbedder`, which says to take the out of the "tokens" `TokenIndexer` and
embed it using fixed, pretrained glove vectors (also provided on S3).  If I wanted to also use a
CNN over character ids, I would need to add a `TokenCharactersIndexer` to the `TokenIndexers` in
the `DatasetReader`, and a corresponding entry to the `TextFieldEmbedder` parameters specifying a
`TokenCharactersEncoder`.

The title and abstract encoders are both bi-directional LSTMs.  The actual LSTM implementation
here is a wrapper around pytorch's built-in LSTMs that make them conform to the `Seq2VecEncoder`
API.  All of the parameters except for "type" get passed directly to pytorch code.

The feed-forward network has a configurable depth, width, and activation.  You can look at the
[documentation](https://allenai.github.io/allennlp-docs/api/allennlp.modules.feedforward.html) of
`FeedForward` for more information on these parameters.

And that's it!  You're now the proud owner of a new `DatasetReader`, `Model`, and means to train
the model on your favorite data.

The only remaining twist is that `allennlp.run` doesn't know about
our custom `Model` and `DatasetReader`, which means we'll need to
provide an extra parameter so that it loads and registers them.

Here our custom code all lives in the `my_library` module,
which means we need to add an extra parameter

```
--include-package my_library
```

which (as long that package is somewhere visible on our PATH)
will load all of the `my_library/...` submodules
(and hence register all of our custom classes)
before training the model.

You can checkout the code from the [github
repository](https://github.com/allenai/allennlp-as-a-library-example/), run the setup commands
mentioned above to install AllenNLP, pytorch, and spacy, and try training this with:

```bash
python -m allennlp.run train \
    experiments/venue_classifier.json \
    -s /tmp/venue_output_dir \
    --include-package my_library
```

When we do this, we get to around 80% validation accuracy after a few epochs of training.
