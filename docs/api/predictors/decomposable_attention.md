# allennlp.predictors.decomposable_attention

## DecomposableAttentionPredictor
```python
DecomposableAttentionPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader) -> None
```

Predictor for the :class:`~allennlp.models.DecomposableAttention` model.

### predict
```python
DecomposableAttentionPredictor.predict(self, premise:str, hypothesis:str) -> Dict[str, Any]
```

Predicts whether the hypothesis is entailed by the premise text.

Parameters
----------
premise : ``str``
    A passage representing what is assumed to be true.

hypothesis : ``str``
    A sentence that may be entailed by the premise.

Returns
-------
A dictionary where the key "label_probs" determines the probabilities of each of
[entailment, contradiction, neutral].

### predictions_to_labeled_instances
```python
DecomposableAttentionPredictor.predictions_to_labeled_instances(self, instance:allennlp.data.instance.Instance, outputs:Dict[str, numpy.ndarray]) -> List[allennlp.data.instance.Instance]
```

This function takes a model's outputs for an Instance, and it labels that instance according
to the output. For example, in classification this function labels the instance according
to the class with the highest probability. This function is used to to compute gradients
of what the model predicted. The return type is a list because in some tasks there are
multiple predictions in the output (e.g., in NER a model predicts multiple spans). In this
case, each instance in the returned list of Instances contains an individual
entity prediction as the label.

