# Creating your Configuration File

AllenNLP experiments are driven by JSON-like configuration files.
One frequent piece of feedback we get is that it's difficult to know
just what needs to go in the configuration file. Accordingly, we've
built a couple of tools (that are still somewhat experimental)
to help you out.

## `allennlp configure`

One option is to use the new `allennlp configure` command, which
produces stubs that can be used to produce configuration files.

If you run it without any options, you get an output corresponding to the top-level configuration:

```bash
$ allennlp configure

configuration stub for AllenNLP:

{
    "dataset_reader": <class 'allennlp.data.dataset_readers.dataset_reader.DatasetReader'> (configurable) // specify your dataset reader here
    // "validation_dataset_reader": <class 'allennlp.data.dataset_readers.dataset_reader.DatasetReader'> (configurable) (default: None ) // same as dataset_reader by default
    "train_data_path": <class 'str'> // path to the training data
    // "validation_data_path": <class 'str'> (default: None ) // path to the validation data
    // "test_data_path": <class 'str'> (default: None ) // path to the test data (you probably don't want to use this!)
    // "evaluate_on_test": <class 'bool'> (default: False ) // whether to evaluate on the test dataset at the end of training (don't do it!
    "model": <class 'allennlp.models.model.Model'> (configurable) // specify your model here
    "iterator": <class 'allennlp.data.iterators.data_iterator.DataIterator'> (configurable) // specify your data iterator here
    "trainer": <class 'allennlp.training.trainer.Trainer'> (configurable) // specify the trainer parameters here
    // "datasets_for_vocab_creation": typing.List[str] (default: None ) // if not specified, use all datasets
    // "vocabulary": <class 'allennlp.data.vocabulary.Vocabulary'> (configurable) (default: None ) // vocabulary options
}
```

Obviously, this is not a valid configuration, but it can easily be turned into one.
The first line indicates that you need a `"dataset_reader"` key, and that its value
should be the configuration for a `DatasetReader` class.

The fourth commented-out line shows that you *can* specify a `"validation_data_path"`
and that it should be a string, but that doing so is optional (hence the commented-out)
and that it will default to `None` if not specified.

To proceed, we would need to create a configuration for the dataset reader:

```bash
$ allennlp configure allennlp.data.dataset_readers.dataset_reader.DatasetReader

DatasetReader is an abstract base class, choose one of the following subclasses:

	 allennlp.data.dataset_readers.conll2003.Conll2003DatasetReader
	 allennlp.data.dataset_readers.coreference_resolution.conll.ConllCorefReader
	 allennlp.data.dataset_readers.coreference_resolution.winobias.WinobiasReader
	 allennlp.data.dataset_readers.language_modeling.LanguageModelingReader
	 allennlp.data.dataset_readers.nlvr.NlvrDatasetReader
	 allennlp.data.dataset_readers.penn_tree_bank.PennTreeBankConstituencySpanDatasetReader
	 allennlp.data.dataset_readers.reading_comprehension.squad.SquadReader
	 allennlp.data.dataset_readers.reading_comprehension.triviaqa.TriviaQaReader
	 allennlp.data.dataset_readers.semantic_role_labeling.SrlReader
	 allennlp.data.dataset_readers.seq2seq.Seq2SeqDatasetReader
	 allennlp.data.dataset_readers.sequence_tagging.SequenceTaggingDatasetReader
	 allennlp.data.dataset_readers.snli.SnliReader
	 allennlp.data.dataset_readers.stanford_sentiment_tree_bank.StanfordSentimentTreeBankDatasetReader
	 allennlp.data.dataset_readers.wikitables.WikiTablesDatasetReader
```

This tells us that we need to pick a specific dataset reader subclass. Let's use the `SquadReader`:

```bash
$ allennlp configure allennlp.data.dataset_readers.reading_comprehension.squad.SquadReader

configuration stub for allennlp.data.dataset_readers.reading_comprehension.squad.SquadReader:

{
    "type": "squad",
    // "tokenizer": <class 'allennlp.data.tokenizers.tokenizer.Tokenizer'> (configurable) (default: None ) // We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
Default is ```WordTokenizer()``.
    // "token_indexers": typing.Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer] (default: None ) // We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
Default is ``{"tokens": SingleIdTokenIndexer()}``.
    // "lazy": <class 'bool'> (default: False )
}
```

This gives us a stub for our dataset reader configuration, and we can continue.

This is sort of an awkward workflow, but you might find it helpful.

## The Configuration Wizard

We have also created a (very experimental) configuration wizard that runs in your browser. To launch it, just run

```
$ python -m allennlp.service.config_explorer
serving Config Explorer on port 8123
```

and then go to [localhost:8123](http://localhost:8123) in your browser.

![configuration wizard](configurator_images/configurator.1.png)

You can see the same fields and annotations as in the command line version.
(The comments are hidden as tooltips, mouse over the question mark button to see them).

Required fields have their names in black, while optional fields have their names in gray.
Required fields that are not completed are highlighted in red.

If you click on the dataset_reader's "CONFIGURE" button, a dropdown will appear:

![configuration wizard](configurator_images/configurator.2.png)

with the same choices as before:

![configuration wizard](configurator_images/configurator.3.png)

and if you select SquadReader again, a configurator for the dataset reader piece will appear:

![configuration wizard](configurator_images/configurator.4.png)

Notice there is also a "X" button to remove the dataset reader configuration if you decide you don't want it.

As you fill in the configuration options, at the bottom of the page an actual JSON configuration
will update in real time:

![configuration wizard](configurator_images/configurator.5.png)

You could, if you like, fill out the entire configuration using this wizard,
copy the generated JSON into a file, and then launch an experiment.

## Caveat

These features are experimental. In particular, the UI for the configuration wizard
should be thought of as a prototype and will certainly improve in the future.
