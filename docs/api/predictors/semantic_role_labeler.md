# allennlp.predictors.semantic_role_labeler

## SemanticRoleLabelerPredictor
```python
SemanticRoleLabelerPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader, language:str='en_core_web_sm') -> None
```

Predictor for the :class:`~allennlp.models.SemanticRoleLabeler` model.

### predict
```python
SemanticRoleLabelerPredictor.predict(self, sentence:str) -> Dict[str, Any]
```

Predicts the semantic roles of the supplied sentence and returns a dictionary
with the results.

.. code-block:: js

    {"words": [...],
     "verbs": [
        {"verb": "...", "description": "...", "tags": [...]},
        ...
        {"verb": "...", "description": "...", "tags": [...]},
    ]}

Parameters
----------
sentence, ``str``
    The sentence to parse via semantic role labeling.

Returns
-------
A dictionary representation of the semantic roles in the sentence.

### predict_tokenized
```python
SemanticRoleLabelerPredictor.predict_tokenized(self, tokenized_sentence:List[str]) -> Dict[str, Any]
```

Predicts the semantic roles of the supplied sentence tokens and returns a dictionary
with the results.

Parameters
----------
tokenized_sentence, ``List[str]``
    The sentence tokens to parse via semantic role labeling.

Returns
-------
A dictionary representation of the semantic roles in the sentence.

### predict_batch_json
```python
SemanticRoleLabelerPredictor.predict_batch_json(self, inputs:List[Dict[str, Any]]) -> List[Dict[str, Any]]
```

Expects JSON that looks like ``[{"sentence": "..."}, {"sentence": "..."}, ...]``
and returns JSON that looks like

.. code-block:: js

    [
        {"words": [...],
         "verbs": [
            {"verb": "...", "description": "...", "tags": [...]},
            ...
            {"verb": "...", "description": "...", "tags": [...]},
        ]},
        {"words": [...],
         "verbs": [
            {"verb": "...", "description": "...", "tags": [...]},
            ...
            {"verb": "...", "description": "...", "tags": [...]},
        ]}
    ]

### predict_json
```python
SemanticRoleLabelerPredictor.predict_json(self, inputs:Dict[str, Any]) -> Dict[str, Any]
```

Expects JSON that looks like ``{"sentence": "..."}``
and returns JSON that looks like

.. code-block:: js

    {"words": [...],
     "verbs": [
        {"verb": "...", "description": "...", "tags": [...]},
        ...
        {"verb": "...", "description": "...", "tags": [...]},
    ]}

