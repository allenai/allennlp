# Creating Your Own Models

Using the included models is fine, but at some point you'll probably want to implement your own models, which is what this tutorial is for.

Generally speaking, in order to implement a new model, you'll need to implement a
[`DatasetReader`](https://allenai.github.io/allennlp-docs/api/allennlp.data.dataset_readers.html)
subclass to read in your datasets and a
[`Model`](https://allenai.github.io/allennlp-docs/api/allennlp.models.model.html)
subclass corresponding to the model you want to implement.
(If there's already a `DatasetReader` for the dataset you want to use,
 of course you can reuse that one.)
In this tutorial we'll also implement a custom PyTorch
[`Module`](http://pytorch.org/docs/master/nn.html#torch.nn.Module),
but you won't need to do that in general.

Our [simple tagger](simple-tagger) model
uses an LSTM to capture dependencies between
the words in the input sentence, but doesn't have a great way
to capture dependencies between the _tags_. This can be a problem
for tasks like [named-entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
where you'd never want to (for example) have a "start of a place" tag followed by an "inside a person" tag.

We'll try to build an NER model that can outperform our simple tagger
on the [CoNLL 2003 dataset](https://www.clips.uantwerpen.be/conll2003/ner/),
which (due to licensing reasons) you'll have to source for yourself.

The simple tagger gets about 88%
[span-based F1](https://allenai.github.io/allennlp-docs/api/allennlp.training.metrics.html#span-based-f1-measure)
on the validation dataset. We'd like to do better.

One way to approach this is to add a [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)
layer at the end of our tagging model.
(If you're not familiar with conditional random fields, [this overview paper](https://arxiv.org/abs/1011.4088)
 is helpful, as is [this PyTorch tutorial](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html).)

The "linear-chain" conditional random field we'll implement has a `num_tags` x `num_tags` matrix of transition costs,
where `transitions[i, j]` represents the likelihood of transitioning
from the `j`-th tag to the `i`-th tag.
In addition to whatever tags we're trying to predict, we'll have special
"start" and "end" tags that we'll stick before and after each sentence
in order to capture the "transition" inherent in being the tag at the
beginning or end of a sentence.

As this is just a component of our model, we'll implement it as a [Module](https://allenai.github.io/allennlp-docs/api/allennlp.modules.html).

## Implementing the CRF Module

To implement a PyTorch module, we just need to inherit from [`torch.nn.Module`](http://pytorch.org/docs/master/nn.html#torch.nn.Module)
and override

```python
    def forward(self, *input):
        ...
```

to compute the log-likelihood of the provided inputs.

To initialize this module, we just need the number of tags.

```python
    def __init__(self, num_tags: int) -> None:
        super().__init__()

        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.randn(num_tags, num_tags))

        # Also need logits for transitioning from "start" state and to "end" state.
        self.start_transitions = torch.nn.Parameter(torch.randn(num_tags))
        self.end_transitions = torch.nn.Parameter(torch.randn(num_tags))
```

I'm not going to get into the exact mechanics of how the log-likelihood is calculated;
you should read the aforementioned overview paper
(and look at our implementation)
if you want the details. The key points are

* the input to this module is a `(sequence_length, num_tags)` tensor of logits
  representing the likelihood of each tag at each position in some sequence
  and a `(sequence_length,)` tensor of gold tags. (In fact, we actually provide
  _batches_ consisting of multiple sequences, but I'm glossing over that detail.)
* The likelihood of producing a certain tag at a certain sequence position depends on both
  the input logits at that position and the transition parameters corresponding to the
  tag at the previous position
* Computing the overall likelihood requires summing across all possible tag sequences,
  but we can use clever dynamic programming tricks to do so efficiently.
* We also add a `viterbi_tags()` method that accepts some input logits,
  gets the transition probabilities, and uses the
  [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
  to compute the most likely sequence of tags for a given input.

## Implementing the CRF Tagger Model

The `CrfTagger` is not terribly different from the `SimpleTagger` model,
so we can take that as a starting point. We need to make the following changes:

* give our model a `crf` attribute containing an appropriately initialized
  `ConditionalRandomField` module
* replace the softmax class probabilities with the Viterbi-generated most likely tags
* replace the softmax + cross-entropy loss function
  with the negative of the CRF log-likelihood

We can then register the new model as `"crf_tagger"`.

## Creating a Dataset Reader

The [CoNLL data](https://www.clips.uantwerpen.be/conll2003/ner/) is formatted like

```
   U.N.         NNP  I-NP  I-ORG
   official     NN   I-NP  O
   Ekeus        NNP  I-NP  I-PER
   heads        VBZ  I-VP  O
   for          IN   I-PP  O
   Baghdad      NNP  I-NP  I-LOC
   .            .    O     O
```

where each line contains a token, a part-of-speech tag, a syntactic chunk tag, and a named-entity tag.
An empty line indicates the end of a sentence, and a line

```
-DOCSTART- -X- O O
```

indicates the end of a document. (Our reader is concerned only with sentences
and doesn't care about documents.)

You can poke at the code yourself, but at a high level we use
[`itertools.groupby`](https://docs.python.org/3/library/itertools.html#itertools.groupby)
to chunk our input into groups of either "dividers" or "sentences".
Then for each sentence we split each row into four columns,
create a `TextField` for the token, and create a `SequenceLabelField`
for the tags (which for us will be the NER tags).

## Creating a Config File

As the `CrfTagger` model is quite similar to the `SimpleTagger` model,
we can get away with a similar configuration file. We need to make only
a couple of changes:

* change the `model.type` to `"crf_tagger"`
* change the `"dataset_reader.type"` to `"conll2003"`
* add a `"dataset_reader.tag_label"` field with value "ner" (to indicate that the NER labels are what we're predicting)

We don't *need* to, but we also make a few other changes

* following [Peters, Ammar, Bhagavatula, and Power 2017](https://www.semanticscholar.org/paper/Semi-supervised-sequence-tagging-with-bidirectiona-Peters-Ammar/73e59cb556351961d1bdd4ab68cbbefc5662a9fc), we use a
  Gated Recurrent Unit (GRU) character encoder
  as well as a GRU for our phrase encoder
* we also start with pretrained [GloVe vectors](https://nlp.stanford.edu/projects/glove/) for our token embeddings
* we add a regularizer that applies a L2 penalty just to the `transitions`
  parameters to help avoid overfitting
* we add a `test_data_path` and set `evaluate_on_test` to true.
  This is mostly to ensure that our token embedding layer loads the GloVe
  vectors corresponding to tokens in the test data set, so that they are not
  treated as out-of-vocabulary at evaluation time. The second flag just evaluates
  the model on the test set when training stops. Use this flag cautiously,
  when you're doing real science you don't want to evaluate on your test set too often.

## Putting It All Together

At this point we're ready to train the model.
In this case our new classes are part of the `allennlp` library,
which means we can just use `python -m allennlp.run train`:

```bash
$ python -m allennlp.run train \
    tutorials/getting_started/crf_tagger.json \
    -s /tmp/crf_model
```

If you were to create your own model outside of
the allennlp codebase, they wouldn't be.

In that case `allennlp.run` needs extra info to load the modules in which
you've defined your classes, they never get registered, and then
AllenNLP is unable to instantiate them based on the configuration file.

You can specify one more more extra packages using the
`--include-packages` flag. For example, imagine that
your model is in the module `myallennlp.model`
and your dataset reader is in the module `myallennlp.dataset_reader`.

Then you would just
```bash
$ python -m allennlp.run train \
    /path/to/your/model/configuration \
    -s /path/to/serialization/dir \
    --include-package myallennlp
```

and (as long as your package is somewhere on the PATH
where Python looks for packages), your custom classes
will all get registered and used correctly.
