# allennlp.predictors.constituency_parser

## ConstituencyParserPredictor
```python
ConstituencyParserPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader, language:str='en_core_web_sm') -> None
```

Predictor for the :class:`~allennlp.models.SpanConstituencyParser` model.

### predict
```python
ConstituencyParserPredictor.predict(self, sentence:str) -> Dict[str, Any]
```

Predict a constituency parse for the given sentence.
Parameters
----------
sentence The sentence to parse.

Returns
-------
A dictionary representation of the constituency tree.

