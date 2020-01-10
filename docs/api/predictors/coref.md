# allennlp.predictors.coref

## CorefPredictor
```python
CorefPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader, language:str='en_core_web_sm') -> None
```

Predictor for the :class:`~allennlp.models.coreference_resolution.CoreferenceResolver` model.

### predict
```python
CorefPredictor.predict(self, document:str) -> Dict[str, Any]
```

Predict the coreference clusters in the given document.

.. code-block:: js

    {
    "document": [tokenised document text]
    "clusters":
      [
        [
          [start_index, end_index],
          [start_index, end_index]
        ],
        [
          [start_index, end_index],
          [start_index, end_index],
          [start_index, end_index],
        ],
        ....
      ]
    }

Parameters
----------
document : ``str``
    A string representation of a document.

Returns
-------
A dictionary representation of the predicted coreference clusters.

### predict_tokenized
```python
CorefPredictor.predict_tokenized(self, tokenized_document:List[str]) -> Dict[str, Any]
```

Predict the coreference clusters in the given document.

Parameters
----------
tokenized_document : ``List[str]``
    A list of words representation of a tokenized document.

Returns
-------
A dictionary representation of the predicted coreference clusters.

### predictions_to_labeled_instances
```python
CorefPredictor.predictions_to_labeled_instances(self, instance:allennlp.data.instance.Instance, outputs:Dict[str, numpy.ndarray]) -> List[allennlp.data.instance.Instance]
```

Takes each predicted cluster and makes it into a labeled ``Instance`` with only that
cluster labeled, so we can compute gradients of the loss `on the model's prediction of that
cluster`.  This lets us run interpretation methods using those gradients.  See superclass
docstring for more info.

### replace_corefs
```python
CorefPredictor.replace_corefs(document:spacy.tokens.doc.Doc, clusters:List[List[List[int]]]) -> str
```

Uses a list of coreference clusters to convert a spacy document into a
string, where each coreference is replaced by its main mention.

### coref_resolved
```python
CorefPredictor.coref_resolved(self, document:str) -> str
```

Produce a document where each coreference is replaced by the its main mention

Parameters
----------
document : ``str``
    A string representation of a document.

Returns
-------
A string with each coference replaced by its main mention

