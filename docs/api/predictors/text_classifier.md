# allennlp.predictors.text_classifier

## TextClassifierPredictor
```python
TextClassifierPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader) -> None
```

Predictor for any model that takes in a sentence and returns
a single class for it.  In particular, it can be used with
the :class:`~allennlp.models.basic_classifier.BasicClassifier` model

### predictions_to_labeled_instances
```python
TextClassifierPredictor.predictions_to_labeled_instances(self, instance:allennlp.data.instance.Instance, outputs:Dict[str, numpy.ndarray]) -> List[allennlp.data.instance.Instance]
```

This function takes a model's outputs for an Instance, and it labels that instance according
to the output. For example, in classification this function labels the instance according
to the class with the highest probability. This function is used to to compute gradients
of what the model predicted. The return type is a list because in some tasks there are
multiple predictions in the output (e.g., in NER a model predicts multiple spans). In this
case, each instance in the returned list of Instances contains an individual
entity prediction as the label.

