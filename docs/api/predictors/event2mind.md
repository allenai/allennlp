# allennlp.predictors.event2mind

## Event2MindPredictor
```python
Event2MindPredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader) -> None
```

Predictor for the :class:`~allennlp.models.event2mind` model.

### predict
```python
Event2MindPredictor.predict(self, source:str) -> Dict[str, Any]
```

Given a source string of some event, returns a JSON dictionary
containing, for each target type, the top predicted sequences as
indices, as tokens and the log probability of each.

The JSON dictionary looks like:

.. code-block:: js

    {
        `${target_type}_top_k_predictions`: [[1, 2, 3], [4, 5, 6], ...],
        `${target_type}_top_k_predicted_tokens`: [["to", "feel", "brave"], ...],
        `${target_type}_top_k_log_probabilities`: [-0.301, -0.046, ...]
    }

By default ``target_type`` can be xreact, oreact and xintent.

