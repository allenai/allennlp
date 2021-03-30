from allennlp.common.util import JsonDict
from allennlp.data import Instance


def get_fields_to_compare(
    inputs: JsonDict, instance: Instance, input_field_to_attack: str
) -> JsonDict:
    """
    Gets a list of the fields that should be checked for equality after an attack is performed.

    # Parameters

    inputs : `JsonDict`
        The input you want to attack, similar to the argument to a Predictor, e.g., predict_json().
    instance : `Instance`
        A labeled instance that is output from json_to_labeled_instances().
    input_field_to_attack : `str`
        The key in the inputs JsonDict you want to attack, e.g., tokens.

    # Returns

    fields : `JsonDict`
        The fields that must be compared for equality.
    """
    # TODO(mattg): this really should live on the Predictor.  We have some messy stuff for, e.g.,
    # reading comprehension models, and the interpret code can't really know about the internals of
    # that (or at least it shouldn't now, and once we split out the reading comprehension repo, it
    # really *can't*).
    fields_to_compare = {
        key: instance[key]
        for key in instance.fields
        if key not in inputs
        and key != input_field_to_attack
        and key != "metadata"
        and key != "output"
    }
    return fields_to_compare


def instance_has_changed(instance: Instance, fields_to_compare: JsonDict):
    if "clusters" in fields_to_compare:
        # Coref needs a special case here, apparently.  I (mattg) am not sure why the check below
        # doesn't catch this case; TODO: look into this.
        original_clusters = set(tuple(x) for x in fields_to_compare["clusters"])
        new_clusters = set(tuple(x) for x in instance["clusters"])  # type: ignore
        return original_clusters != new_clusters
    if any(instance[field] != fields_to_compare[field] for field in fields_to_compare):
        return True
    return False
