# allennlp.predictors.biaffine_dependency_parser

## BiaffineDependencyParserPredictor
```python
BiaffineDependencyParserPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader, language:str='en_core_web_sm') -> None
```

Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model.

### predict
```python
BiaffineDependencyParserPredictor.predict(self, sentence:str) -> Dict[str, Any]
```

Predict a dependency parse for the given sentence.
Parameters
----------
sentence The sentence to parse.

Returns
-------
A dictionary representation of the dependency tree.

