# allennlp.interpret.attackers.attacker

## Attacker
```python
Attacker(self, predictor:allennlp.predictors.predictor.Predictor) -> None
```

An ``Attacker`` will modify an input (e.g., add or delete tokens) to try to change an AllenNLP
Predictor's output in a desired manner (e.g., make it incorrect).

### initialize
```python
Attacker.initialize(self)
```

Initializes any components of the Attacker that are expensive to compute, so that they are
not created on __init__().  Default implementation is ``pass``.

### attack_from_json
```python
Attacker.attack_from_json(self, inputs:Dict[str, Any], input_field_to_attack:str, grad_input_field:str, ignore_tokens:List[str], target:Dict[str, Any]) -> Dict[str, Any]
```

This function finds a modification to the input text that would change the model's
prediction in some desired manner (e.g., an adversarial attack).

Parameters
----------
inputs : ``JsonDict``
    The input you want to attack (the same as the argument to a Predictor, e.g.,
    predict_json()).
input_field_to_attack : ``str``
    The key in the inputs JsonDict you want to attack, e.g., ``tokens``.
grad_input_field : ``str``
    The field in the gradients dictionary that contains the input gradients.  For example,
    `grad_input_1` will be the field for single input tasks. See get_gradients() in
    `Predictor` for more information on field names.
target : ``JsonDict``
    If given, this is a `targeted` attack, trying to change the prediction to a particular
    value, instead of just changing it from its original prediction.  Subclasses are not
    required to accept this argument, as not all attacks make sense as targeted attacks.
    Perhaps that means we should make the API more crisp, but adding another class is not
    worth it.

Returns
-------
reduced_input : ``JsonDict``
    Contains the final, sanitized input after adversarial modification.

