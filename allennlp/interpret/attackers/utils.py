from allennlp.common.util import JsonDict
from allennlp.data import Instance

def get_fields_to_compare(inputs: JsonDict, instance: Instance, input_field_to_attack: str) -> JsonDict:
    """
    Gets a list of the fields that should be checked for equality after an attack is performed.

    Parameters
    ----------
    inputs : ``JsonDict``
        The input you want to attack, similar to the argument to a Predictor, e.g., predict_json().
    instance : ``Instance``
        A labeled instance that is output from json_to_labeled_instances().
    input_field_to_attack : ``str``
        The key in the inputs JsonDict you want to attack, e.g., `tokens`.

    Returns
    -------
    fields : ``JsonDict``
        The fields that must be compared for equality.
    """
    fields_to_compare = {
            key: instance[key]
            for key in instance.fields
            if key not in inputs and key != input_field_to_attack and key != 'metadata'
    }
    return fields_to_compare
