# allennlp.interpret.saliency_interpreters.saliency_interpreter

## SaliencyInterpreter
```python
SaliencyInterpreter(self, predictor:allennlp.predictors.predictor.Predictor) -> None
```

A ``SaliencyInterpreter`` interprets an AllenNLP Predictor's outputs by assigning a saliency
score to each input token.

### saliency_interpret_from_json
```python
SaliencyInterpreter.saliency_interpret_from_json(self, inputs:Dict[str, Any]) -> Dict[str, Any]
```

This function finds a modification to the input text that would change the model's
prediction in some desired manner (e.g., an adversarial attack).

Parameters
----------
inputs : ``JsonDict``
    The input you want to interpret (the same as the argument to a Predictor, e.g., predict_json()).

Returns
-------
interpretation : ``JsonDict``
    Contains the normalized saliency values for each input token. The dict has entries for
    each instance in the inputs JsonDict, e.g., ``{instance_1: ..., instance_2:, ... }``.
    Each one of those entries has entries for the saliency of the inputs, e.g.,
    ``{grad_input_1: ..., grad_input_2: ... }``.

