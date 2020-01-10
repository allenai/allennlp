# allennlp.interpret.attackers.utils

## get_fields_to_compare
```python
get_fields_to_compare(inputs:Dict[str, Any], instance:allennlp.data.instance.Instance, input_field_to_attack:str) -> Dict[str, Any]
```

Gets a list of the fields that should be checked for equality after an attack is performed.

Parameters
----------
inputs : ``JsonDict``
    The input you want to attack, similar to the argument to a Predictor, e.g., predict_json().
instance : ``Instance``
    A labeled instance that is output from json_to_labeled_instances().
input_field_to_attack : ``str``
    The key in the inputs JsonDict you want to attack, e.g., tokens.

Returns
-------
fields : ``JsonDict``
    The fields that must be compared for equality.

