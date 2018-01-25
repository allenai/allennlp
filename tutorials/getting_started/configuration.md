# Configuring Experiments

Now that we know how to train and evaluate models,
let's take a deeper look at our experiment configuration file,
[tutorials/getting_started/simple_tagger.json](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/simple_tagger.json).

The configuration is a [HOCON](https://github.com/typesafehub/config/blob/master/HOCON.md) file
that defines all the parameters for our experiment and model. Don't worry if you're not familiar
with HOCON, any JSON file is valid HOCON; indeed, the configuration file we use in this tutorial
is just JSON.

In this tutorial we'll go through
each section of the configuration file in detail, explaining what all the parameters mean.

## A preliminary: Registrable and from_params

Most AllenNLP classes inherit from the
[`Registrable`](http://docs.allennlp.org/en/latest/api/allennlp.common.html#allennlp.common.registrable.Registrable)
base class,
which gives them a named registry for their subclasses. This means that if we had
a `Model(Registrable)` base class ([we do](http://docs.allennlp.org/en/latest/api/allennlp.models.html#allennlp.models.model.Model)),
and we decorated a subclass like

```python
@Model.register("custom")
class CustomModel(Model):
    ...
```

then we would be able to recover the `CustomModel` class using

```
Model.by_name("custom")
```

By convention, all such classes have a `from_params` factory method that
allows you to instantiate instances from a
[`Params`](http://docs.allennlp.org/en/latest/api/allennlp.common.html#allennlp.common.params.Params)
object, which is basically a `dict` of parameters
with some added functionality that we won't get into here.

This is how AllenNLP is able to use configuration files to instantiate the objects it needs.
It can do (in essence):

```python
# Grab the part of the `config` that defines the model
model_params = config.pop("model")

# Find out which model subclass we want
model_name = model_params.pop("type")

# Instantiate that subclass with the remaining model params
model = Model.by_name(model_name).from_params(model_params)
```

Because a class doesn't get registered until it's loaded, any code that uses
`BaseClass.by_name('subclass_name')` must have already imported the code for `Subclass`.
In particular, this means that once you start creating your own named models and helper classes,
the included `allennlp.run` command will not be aware of them. However, `allennlp.run` is simply
a wrapper around the `allennlp.commands.main` function,
which means you just need to create your own script
that imports all of your custom classes and then calls `allennlp.commands.main()`.

## Datasets and Instances and Fields

We train and evaluate our models on `Dataset`s. A
[`Dataset`](http://docs.allennlp.org/en/latest/api/allennlp.data.html#allennlp.data.dataset.Dataset)
is a collection of
[`Instance`](http://docs.allennlp.org/en/latest/api/allennlp.data.html#allennlp.data.instance.Instance)s.
In our tagging experiment,
each dataset is a collection of tagged sentences, and each instance is one of those tagged sentences.

An instance consists of
[`Field`](http://docs.allennlp.org/en/latest/api/allennlp.data.fields.html#allennlp.data.fields.field.Field)s,
each of which represents some part of the instance as arrays suitable for feeding into a model.

In our tagging setup, each instance will contain a
[`TextField`](http://docs.allennlp.org/en/latest/api/allennlp.data.fields.html#allennlp.data.fields.text_field.TextField)
representing the words/tokens of the sentence and a
[`SequenceLabelField`](http://docs.allennlp.org/en/latest/api/allennlp.data.fields.html#allennlp.data.fields.sequence_label_field.SequenceLabelField)
representing the corresponding part-of-speech tags.

How do we turn a text file full of sentences into a `Dataset`? With a
[`DatasetReader`](http://docs.allennlp.org/en/latest/api/allennlp.data.dataset_readers.html#allennlp.data.dataset_readers.dataset_reader.DatasetReader)
specified by our configuration file.

## DatasetReaders

The first section of our configuration file defines the `dataset_reader`:

```js
  "dataset_reader": {
    "type": "sequence_tagging",
    "word_tag_delimiter": "/",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters"
      }
    }
  },

```

Here we've specified that we want to use the `DatasetReader` subclass that's registered
under the name `"sequence_tagging"`. Unsurprisingly, this is the
[`SequenceTaggingDatasetReader`](http://docs.allennlp.org/en/latest/api/allennlp.data.dataset_readers.html#allennlp.data.dataset_readers.sequence_tagging.SequenceTaggingDatasetReader)
subclass. This reader assumes a text file of newline-separated sentences, where each sentence looks like

```
word1{wtd}tag1{td}word2{wtd}tag2{td}...{td}wordn{wtd}tagn
```

For some "word tag delimiter" `{wtd}` and some "token delimiter" `{td}`.

_Our_ data files look like

```
The/at detectives/nns placed/vbd Barco/np under/in arrest/nn
```

which is why we need to specify

```js
    "word_tag_delimiter": "/",
```

We don't need to specify anything for the "token delimiter",
since the default split-on-whitespace behavior is already correct.

If you look at the code for `SequenceTaggingDatasetReader.read()`,
it turns each sentence into a `TextField`
of tokens and a `SequenceLabelField` of tags. The latter isn't
really configurable, but the former wants a dictionary of
[TokenIndexer](http://docs.allennlp.org/en/latest/api/allennlp.data.token_indexers.html#allennlp.data.token_indexers.token_indexer.TokenIndexer)s
that indicate how to convert the tokens into arrays.

Our configuration specifies two token indexers:

```js
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters"
      }
    }
```

The first, `"tokens"`, is a
[`SingleIdTokenIndexer`](http://docs.allennlp.org/en/latest/api/allennlp.data.token_indexers.html#allennlp.data.token_indexers.single_id_token_indexer.SingleIdTokenIndexer)
that just represents each token (word) as a single integer.
The configuration also specifies that we *lowercase* the tokens before encoding;
that is, that this token indexer should ignore case.

The second, `"token_characters"`, is a
[`TokenCharactersIndexer`](http://docs.allennlp.org/en/latest/api/allennlp.data.token_indexers.html#allennlp.data.token_indexers.token_characters_indexer.TokenCharactersIndexer)
that represents each token as a list of int-encoded characters.

Notice that this gives us two different encodings for each token.
Each encoding has a name, in this case `"tokens"` and `"token_characters"`,
and these names will be referenced later by the model.

## Training and Validation Data

The next section specifies the data to train and validate the model on:

```js
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.train",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/getting-started/sentences.small.dev",
```

They can be specified either as local paths on your machine or as URLs to files hosted on, for example, Amazon S3.
In the latter case, AllenNLP will cache (and reuse) the downloaded files in `~/.allennlp/datasets`, using the
[ETag](https://en.wikipedia.org/wiki/HTTP_ETag) to determine when to download a new version.

## The Model

The next section configures our model.

```js
  "model": {
    "type": "simple_tagger",
```

This indicates we want to use the `Model` subclass that's registered as `"simple_tagger"`,
which is the
[`SimpleTagger`](http://docs.allennlp.org/en/latest/api/allennlp.models.html#allennlp.models.simple_tagger.SimpleTagger) model.

If you look at its code, you'll see it consists of a
[`TextFieldEmbedder`](http://docs.allennlp.org/en/latest/api/allennlp.modules.text_field_embedders.html#allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder)
that embeds the output of our text fields, a
[`Seq2SeqEncoder`](http://docs.allennlp.org/en/latest/api/allennlp.modules.seq2seq_encoders.html#allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder)
that transforms that sequence into an output sequence,
and a linear layer to convert the output sequences
into logits representing the probabilities of predicted tags.
(The last layer is not configurable and so won't appear in our configuration file.)

### The Text Field Embedder

Let's first look at the text field embedder configuration:

```js
    "text_field_embedder": {
            "tokens": {
                    "type": "embedding",
                    "embedding_dim": 50
            },
            "token_characters": {
              "type": "character_encoding",
              "embedding": {
                "embedding_dim": 8
              },
              "encoder": {
                "type": "cnn",
                "embedding_dim": 8,
                "num_filters": 50,
                "ngram_filter_sizes": [5]
              },
              "dropout": 0.2
            }
    },
```

You can see that it has an entry for each of the named encodings in our `TextField`.
Each entry specifies a
[`TokenEmbedder`](http://docs.allennlp.org/en/latest/api/allennlp.modules.token_embedders.html?highlight=embedding#allennlp.modules.token_embedders.token_embedder.TokenEmbedder)
that indicates how to embed
the tokens encoded with that name. The `TextFieldEmbedder`'s output is the concatenation
of these embeddings.

The `"tokens"` input (which consists of integer encodings of the lowercased words in the input)
gets fed into an
[`Embedding`](http://docs.allennlp.org/en/latest/api/allennlp.modules.token_embedders.html?highlight=embedding#allennlp.modules.token_embedders.embedding.Embedding)
module that embeds the vocabulary words in a 50-dimensional space, as specified by the `embedding_dim` parameter.

The `"token_characters"` input (which consists of integer-sequence encodings of the characters in each word)
gets fed into a
[`TokenCharactersEncoder`](http://docs.allennlp.org/en/latest/api/allennlp.modules.token_embedders.html?highlight=embedding#allennlp.modules.token_embedders.token_characters_encoder.TokenCharactersEncoder),
which embeds the characters in an 8-dimensional space
and then applies a
[`CnnEncoder`](http://docs.allennlp.org/en/latest/api/allennlp.modules.seq2vec_encoders.html#allennlp.modules.seq2vec_encoders.cnn_encoder.CnnEncoder)
that uses 50 filters and so also produces a 50-dimensional output. You can see that this encoder also uses a 20% dropout during training.

The output of this `TextFieldEmbedder` is a 50-dimensional vector for `"tokens"`
concatenated with a 50-dimensional vector for `"token_characters`";
that is, a 100-dimensional vector.

Because both the encoding of `TextFields` and the `TextFieldEmbedder` are configurable in this way,
it is trivial to experiment with different word representations as input to your model, switching
between simple word embeddings, word embeddings concatenated with a character-level CNN, or even
using a pre-trained model to get word-in-context embeddings, without changing a single line of
code.

### The Seq2SeqEncoder

The output of the `TextFieldEmbedder` is processed by the "stacked encoder",
which needs to be a
[`Seq2SeqEncoder`](http://docs.allennlp.org/en/latest/api/allennlp.modules.seq2seq_encoders.html#allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder):

```js
    "stacked_encoder": {
            "type": "lstm",
            "input_size": 100,
            "hidden_size": 100,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": true
    }
```

Here the `"lstm"` encoder is just a thin wrapper around
[`torch.nn.LSTM`](http://pytorch.org/docs/master/nn.html#torch.nn.LSTM),
and its parameters are simply passed through to the PyTorch constructor.
Its input size needs to match the 100-dimensional output size of the
previous embedding layer.

And, as mentioned above, the output of this layer gets passed to a linear layer
that doesn't need any configuration. That's all for the model.

## Training the Model

The rest of the config file is dedicated to the training process.

```js
  "iterator": {"type": "basic", "batch_size": 32},
```

We'll iterate over our datasets using a
[`BasicIterator`](http://docs.allennlp.org/en/latest/api/allennlp.data.iterators.html#allennlp.data.iterators.basic_iterator.BasicIterator)
that pads our data and processes it in batches of size 32.


```js
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1
  }
}
```

Finally, we'll optimize using
[`torch.optim.Adam`](http://pytorch.org/docs/master/optim.html#torch.optim.Adam)
with its default parameters;
we'll run the training for 40 epochs;
we'll stop prematurely if we get no improvement for 10 epochs;
and we'll train on the CPU.  If you wanted to train on a GPU,
you'd change `cuda_device` to its device id.
If you have just one GPU that should be `0`.

That's our entire experiment configuration. If we want to change our optimizer,
our batch size, our embedding dimensions, or any other hyperparameters,
all we need to do is modify this config file and `train` another model.

The training configuration is always saved as part of the model archive,
which means that you can always see how a saved model was trained.


### Next Steps

Continue on to our [Creating a Model](creating_a_model.md) tutorial.
