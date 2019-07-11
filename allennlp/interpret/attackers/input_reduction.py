# pylint: disable=dangerous-default-value
from typing import List, Tuple
import numpy as np
import torch
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance

@Attacker.register('input-reduction')
class InputReduction(Attacker):
    """
    Runs the input reduction method from `Pathologies of Neural Models Make Interpretations
    Difficult` https://arxiv.org/abs/1804.07781, which removes as many words as possible
    from the input without changing the model's prediction.
    """
    def attack_from_json(self, inputs: JsonDict,
                         input_field_to_attack: str = 'tokens',
                         grad_input_field: str = 'grad_input_1',
                         ignore_tokens: List[str] = ["@@NULL@@"]):
        original_instances = self.predictor.json_to_labeled_instances(inputs)
        final_tokens = []
        fields_to_check = {}
        for current_instance in original_instances:
            current_instances = [current_instance] # type: List[Instance]
            original_tokens = getattr(current_instances[0][input_field_to_attack], 'tokens')

            # Save fields that must be checked for equality
            test_instances = self.predictor.json_to_labeled_instances(inputs)
            for key in current_instances[0].fields:
                if key not in inputs and key != input_field_to_attack:
                    fields_to_check[key] = test_instances[0][key]

            # Set num_ignore_tokens, which tells input reduction when to stop
            num_ignore_tokens = 0
            # Keep at least one token for classification/entailment/etc.
            if "tags" not in current_instances[0]:
                num_ignore_tokens = 1

            # Set num_ignore_tokens for NER and build token mask
            else:
                num_ignore_tokens, tag_mask, original_tags = \
                    get_ner_tags_and_mask(current_instances, input_field_to_attack, ignore_tokens)
            current_tokens = getattr(current_instances[0][input_field_to_attack], 'tokens')
            smallest_idx = -1
            # keep removing tokens
            while len(current_instances[0][input_field_to_attack]) >= num_ignore_tokens: # type: ignore
                # get gradients and predictions
                grads, outputs = self.predictor.get_gradients(current_instances)
                for output in outputs:
                    if isinstance(outputs[output], torch.Tensor):
                        outputs[output] = outputs[output].detach().cpu().numpy().squeeze().squeeze()
                    elif isinstance(outputs[output], list):
                        outputs[output] = outputs[output][0]

                # Check if any fields have changed, if so, break loop
                if "tags" not in current_instances[0]:
                    equal = True
                    for field in fields_to_check:
                        if field in current_instances[0].fields:
                            equal = equal and current_instances[0][field].__eq__(fields_to_check[field])
                        else:
                            equal = equal and outputs[field] == fields_to_check[field]
                    if not equal:
                        break

                # special case for sentence tagging (we have tested NER)
                else:
                    if smallest_idx != -1:
                        del tag_mask[smallest_idx]
                    cur_tags = [outputs["tags"][x] for x in range(len(outputs["tags"])) if tag_mask[x]]
                    if cur_tags != original_tags:
                        break

                # remove a token from the input
                current_tokens = getattr(current_instances[0][input_field_to_attack], 'tokens')
                current_instances, smallest_idx = \
                    remove_one_token(grads[grad_input_field],
                                     current_instances,
                                     input_field_to_attack, ignore_tokens)

            final_tokens.append(current_tokens)
        return sanitize({"final": final_tokens, "original": original_tokens})

def remove_one_token(grads: np.ndarray,
                     instances: List[Instance] = None,
                     input_field_to_attack: str = 'tokens',
                     ignore_tokens: List[str] = ["@@NULL@@"]) -> Tuple[List[Instance], int]:
    """
    Finds the token with the smallest gradient and removes it.
    """
    # Compute L2 norm of all grads.
    grads_mag = [np.sqrt(grad.dot(grad)) for grad in grads]

    # Skip all ignore_tokens by setting grad to infinity
    field = instances[0][input_field_to_attack]
    tokens = getattr(field, 'tokens')
    for tok_idx, tok in enumerate(tokens):
        if tok in ignore_tokens:
            grads_mag[tok_idx] = float("inf")

    # For NER, skip all tokens that are not in outside
    if "tags" in instances[0]:
        field_list = getattr(instances[0]["tags"], 'field_list')
        for idx, label in enumerate(field_list):
            if label.label != "O":
                grads_mag[idx] = float("inf")

    smallest = np.argmin(grads_mag)
    if smallest == float("inf"): # if all are ignored tokens, return.
        return instances, smallest

    # remove smallest
    input_field = instances[0][input_field_to_attack]
    inputs_before_smallest = getattr(input_field, 'tokens')[0:smallest]
    inputs_after_smallest = getattr(input_field, 'tokens')[smallest + 1:]
    setattr(input_field, 'tokens', inputs_before_smallest + inputs_after_smallest)

    if "tags" in instances[0]:
        field_list = getattr(instances[0]["tags"], 'field_list')
        field_list_before_smallest = getattr(instances[0]["tags"], 'field_list')[0:smallest]
        field_list_after_smallest = getattr(instances[0]["tags"], 'field_list')[smallest + 1:]
        setattr(instances[0]["tags"], 'field_list', field_list_before_smallest + field_list_after_smallest)

    instances[0].indexed = False
    return instances, smallest

def get_ner_tags_and_mask(current_instances: List[Instance] = None,
                          input_field_to_attack: str = 'tokens',
                          ignore_tokens: List[str] = ["@@NULL@@"]):
    """
    Used for the NER task. Sets the num_ignore tokens, saves the original
     predicted tag and a 0/1 mask in the position of the tags
    """
    # Set num_ignore_tokens
    num_ignore_tokens = 0
    input_field = current_instances[0][input_field_to_attack]
    tokens = getattr(input_field, 'tokens')
    for token in tokens:
        if str(token) in ignore_tokens:
            num_ignore_tokens += 1

    # save the original tags and a 0/1 mask where the tags are
    tag_mask = []
    original_tags = []
    tag_field = current_instances[0]["tags"]
    for label in getattr(tag_field, 'field_list'):
        if label.label != "O":
            tag_mask.append(1)
            original_tags.append(label.label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)
    return num_ignore_tokens, tag_mask, original_tags
