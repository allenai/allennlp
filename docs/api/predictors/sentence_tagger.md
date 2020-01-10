# allennlp.predictors.sentence_tagger

## SentenceTaggerPredictor
```python
SentenceTaggerPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader, language:str='en_core_web_sm') -> None
```

Predictor for any model that takes in a sentence and returns
a single set of tags for it.  In particular, it can be used with
the :class:`~allennlp.models.crf_tagger.CrfTagger` model
and also
the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.

### predictions_to_labeled_instances
```python
SentenceTaggerPredictor.predictions_to_labeled_instances(self, instance:allennlp.data.instance.Instance, outputs:Dict[str, numpy.ndarray]) -> List[allennlp.data.instance.Instance]
```

This function currently only handles BIOUL tags.

Imagine an NER model predicts three named entities (each one with potentially
multiple tokens). For each individual entity, we create a new Instance that has
the label set to only that entity and the rest of the tokens are labeled as outside.
We then return a list of those Instances.

For example:
Mary  went to Seattle to visit Microsoft Research
U-Per  O    O   U-Loc  O   O     B-Org     L-Org

We create three instances.
Mary  went to Seattle to visit Microsoft Research
U-Per  O    O    O     O   O       O         O

Mary  went to Seattle to visit Microsoft Research
O      O    O   U-LOC  O   O       O         O

Mary  went to Seattle to visit Microsoft Research
O      O    O    O     O   O     B-Org     L-Org

