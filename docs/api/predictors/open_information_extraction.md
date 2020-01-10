# allennlp.predictors.open_information_extraction

## join_mwp
```python
join_mwp(tags:List[str]) -> List[str]
```

Join multi-word predicates to a single
predicate ('V') token.

## make_oie_string
```python
make_oie_string(tokens:List[allennlp.data.tokenizers.token.Token], tags:List[str]) -> str
```

Converts a list of model outputs (i.e., a list of lists of bio tags, each
pertaining to a single word), returns an inline bracket representation of
the prediction.

## get_predicate_indices
```python
get_predicate_indices(tags:List[str]) -> List[int]
```

Return the word indices of a predicate in BIO tags.

## get_predicate_text
```python
get_predicate_text(sent_tokens:List[allennlp.data.tokenizers.token.Token], tags:List[str]) -> str
```

Get the predicate in this prediction.

## predicates_overlap
```python
predicates_overlap(tags1:List[str], tags2:List[str]) -> bool
```

Tests whether the predicate in BIO tags1 overlap
with those of tags2.

## get_coherent_next_tag
```python
get_coherent_next_tag(prev_label:str, cur_label:str) -> str
```

Generate a coherent tag, given previous tag and current label.

## merge_overlapping_predictions
```python
merge_overlapping_predictions(tags1:List[str], tags2:List[str]) -> List[str]
```

Merge two predictions into one. Assumes the predicate in tags1 overlap with
the predicate of tags2.

## consolidate_predictions
```python
consolidate_predictions(outputs:List[List[str]], sent_tokens:List[allennlp.data.tokenizers.token.Token]) -> Dict[str, List[str]]
```

Identify that certain predicates are part of a multiword predicate
(e.g., "decided to run") in which case, we don't need to return
the embedded predicate ("run").

## sanitize_label
```python
sanitize_label(label:str) -> str
```

Sanitize a BIO label - this deals with OIE
labels sometimes having some noise, as parentheses.

## OpenIePredictor
```python
OpenIePredictor(self, model:allennlp.models.model.Model, dataset_reader:allennlp.data.dataset_readers.dataset_reader.DatasetReader) -> None
```

Predictor for the :class: `models.SemanticRolelabeler` model (in its Open Information variant).
Used by online demo and for prediction on an input file using command line.

### predict_json
```python
OpenIePredictor.predict_json(self, inputs:Dict[str, Any]) -> Dict[str, Any]
```

Create instance(s) after predicting the format. One sentence containing multiple verbs
will lead to multiple instances.

Expects JSON that looks like ``{"sentence": "..."}``

Returns a JSON that looks like

.. code-block:: js

    {"tokens": [...],
     "tag_spans": [{"ARG0": "...",
                    "V": "...",
                    "ARG1": "...",
                     ...}]}

