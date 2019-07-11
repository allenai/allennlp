from typing import List, Tuple
import numpy as np
import torch
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.interpret.attackers import utils
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance

@Attacker.register('input-reduction')
class InputReduction(Attacker):
    """
    Runs the input reduction method from `Pathologies of Neural Models Make Interpretations
    Difficult` https://arxiv.org/abs/1804.07781, which removes as many words as possible
    from the input without changing the model's prediction.
    """
    def attack_from_json(self, inputs: JsonDict = None,
                         input_field_to_attack: str = 'tokens',
                         grad_input_field: str = 'grad_input_1',
                         ignore_tokens: List[str] = ["@@NULL@@"]):
        original_instances = self.predictor.json_to_labeled_instances(inputs)
        original_tokens = list(original_instances[0][input_field_to_attack])
        final_tokens = []
        for current_instance in original_instances:
            # Save fields that must be checked for equality
            fields_to_compare = utils.get_fields_to_compare(inputs, current_instance, input_field_to_attack)

            # Set num_ignore_tokens, which tells input reduction when to stop
            num_ignore_tokens = 0
            # Keep at least one token for classification/entailment/etc.
            if "tags" not in current_instance:
                num_ignore_tokens = 1

            # Set num_ignore_tokens for NER and build token mask
            else:
                num_ignore_tokens, tag_mask, original_tags = \
                    get_ner_tags_and_mask(current_instance, input_field_to_attack, ignore_tokens)
            current_tokens = list(current_instance[input_field_to_attack])
            smallest_idx = -1

            # keep removing tokens until prediction is about to change
            while len(current_instance[input_field_to_attack]) >= num_ignore_tokens: # type: ignore
                # get gradients and predictions
                grads, outputs = self.predictor.get_gradients([current_instance])
                for output in outputs:
                    if isinstance(outputs[output], torch.Tensor):
                        outputs[output] = outputs[output].detach().cpu().numpy().squeeze().squeeze()
                    elif isinstance(outputs[output], list):
                        outputs[output] = outputs[output][0]

                # Check if any fields have changed, if so, break loop
                if "tags" not in current_instance:
                    if any(current_instance[field] != fields_to_compare[field] for field in fields_to_compare):
                        break

                # special case for sentence tagging (we have tested NER)
                else:
                    if smallest_idx != -1:
                        del tag_mask[smallest_idx]
                    cur_tags = [outputs["tags"][x] for x in range(len(outputs["tags"])) if tag_mask[x]]
                    if cur_tags != original_tags:
                        break

                # remove a token from the input
                current_tokens = list(current_instance[input_field_to_attack])
                current_instance, smallest_idx = \
                    remove_one_token(grads[grad_input_field],
                                     current_instance,
                                     input_field_to_attack, ignore_tokens)

            final_tokens.append(current_tokens)
        return sanitize({"final": final_tokens, "original": original_tokens})

def remove_one_token(grads: np.ndarray = None,
                     instance: Instance = None,
                     input_field_to_attack: str = 'tokens',
                     ignore_tokens: List[str] = ["@@NULL@@"]) -> Tuple[Instance, int]:
    """
    Finds the token with the smallest gradient and removes it.
    """
    # Compute L2 norm of all grads.
    grads_mag = [np.sqrt(grad.dot(grad)) for grad in grads]

    # Skip all ignore_tokens by setting grad to infinity
    field = instance[input_field_to_attack]
    for tok_idx, tok in enumerate(list(field)):
        if tok in ignore_tokens:
            grads_mag[tok_idx] = float("inf")

    # For NER, skip all tokens that are not in outside
    if "tags" in instance:
        field_list = getattr(instance["tags"], 'field_list')
        for idx, label in enumerate(field_list):
            if label.label != "O":
                grads_mag[idx] = float("inf")

    smallest = np.argmin(grads_mag)
    if smallest == float("inf"): # if all are ignored tokens, return.
        return instance, smallest

    # remove smallest
    input_field = instance[input_field_to_attack]
    inputs_before_smallest = list(input_field)[0:smallest]
    inputs_after_smallest = list(input_field)[smallest + 1:]
    setattr(input_field, 'tokens', inputs_before_smallest + inputs_after_smallest)

    if "tags" in instance:
        field_list = getattr(instance["tags"], 'field_list')
        field_list_before_smallest = getattr(instance["tags"], 'field_list')[0:smallest]
        field_list_after_smallest = getattr(instance["tags"], 'field_list')[smallest + 1:]
        setattr(instance["tags"], 'field_list', field_list_before_smallest + field_list_after_smallest)

    instance.indexed = False
    return instance, smallest

def get_ner_tags_and_mask(current_instance: Instance = None,
                          input_field_to_attack: str = 'tokens',
                          ignore_tokens: List[str] = ["@@NULL@@"]):
    """
    Used for the NER task. Sets the num_ignore tokens, saves the original
     predicted tag and a 0/1 mask in the position of the tags
    """
    # Set num_ignore_tokens
    num_ignore_tokens = 0
    input_field = current_instance[input_field_to_attack]
    for token in list(input_field):
        if str(token) in ignore_tokens:
            num_ignore_tokens += 1

    # save the original tags and a 0/1 mask where the tags are
    tag_mask = []
    original_tags = []
    tag_field = current_instance["tags"]
    for label in getattr(tag_field, 'field_list'):
        if label.label != "O":
            tag_mask.append(1)
            original_tags.append(label.label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)
    return num_ignore_tokens, tag_mask, original_tags
