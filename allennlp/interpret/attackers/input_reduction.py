from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch

from allennlp.interpret.attackers.attacker import Attacker
from allennlp.interpret.attackers import utils
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.fields import ListField, TextField, LabelField, SequenceField


@Attacker.register('input-reduction')
class InputReduction(Attacker):
    """
    Runs the input reduction method from `Pathologies of Neural Models Make Interpretations
    Difficult` https://arxiv.org/abs/1804.07781, which removes as many words as possible
    from the input without changing the model's prediction.

    The functions on this class handle a special case for NER by looking for a field called "tags"
    This check is brittle, i.e., the code could break if the name of this field has changed.
    """
    def attack_from_json(self, inputs: JsonDict = None,
                         input_field_to_attack: str = 'tokens',
                         grad_input_field: str = 'grad_input_1',
                         ignore_tokens: List[str] = None):
        ignore_tokens = ["@@NULL@@"] if ignore_tokens is None else ignore_tokens
        original_instances = self.predictor.json_to_labeled_instances(inputs)
        original_text_field: TextField = original_instances[0][input_field_to_attack]  # type: ignore
        original_tokens = deepcopy(original_text_field.tokens)
        final_tokens = []
        for current_instance in original_instances:
            # Save fields that must be checked for equality
            fields_to_compare = utils.get_fields_to_compare(inputs, current_instance, input_field_to_attack)

            # Set num_ignore_tokens, which tells input reduction when to stop
            # We keep at least one token for input reduction on classification/entailment/etc.
            if "tags" not in current_instance:
                num_ignore_tokens = 1

            # Set num_ignore_tokens for NER and build token mask
            else:
                num_ignore_tokens, tag_mask, original_tags = get_ner_tags_and_mask(current_instance,
                                                                                   input_field_to_attack,
                                                                                   ignore_tokens)

            current_text_field: TextField = current_instance[input_field_to_attack]  # type: ignore
            current_tokens = deepcopy(current_text_field.tokens)
            smallest_idx = -1
            # keep removing tokens until prediction is about to change
            while len(current_text_field) >= num_ignore_tokens:
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
                        print("delete tag mask)")
                        del tag_mask[smallest_idx]
                    cur_tags = [outputs["tags"][x] for x in range(len(outputs["tags"])) if tag_mask[x]]
                    if cur_tags != original_tags:
                        break

                # remove a token from the input
                current_tokens = deepcopy(current_text_field.tokens)
                print(current_instance)
                current_instance, smallest_idx = \
                    remove_one_token(current_instance,
                                     input_field_to_attack,
                                     grads[grad_input_field],
                                     ignore_tokens)
                print(current_instance)

            final_tokens.append(current_tokens)
        return sanitize({"final": final_tokens, "original": original_tokens})

def remove_one_token(_instance: Instance,
                     _input_field_to_attack: str,
                     _grads: np.ndarray,
                     _ignore_tokens: List[str]) -> Tuple[Instance, int]:
    """
    Finds the token with the smallest gradient and removes it.
    """
    # Compute L2 norm of all grads.
    grads_mag = [np.sqrt(grad.dot(grad)) for grad in _grads]

    # Skip all ignore_tokens by setting grad to infinity
    text_field: TextField = _instance[_input_field_to_attack]  # type: ignore
    for tok_idx, tok in enumerate(text_field.tokens):
        if tok in _ignore_tokens:
            grads_mag[tok_idx] = float("inf")

    # For NER, skip all tokens that are not in outside
    if "tags" in _instance:
        tag_field: SequenceLabelField = _instance["tags"] # type: ignore
        for idx, label in enumerate(tag_field):
            if label != "O":
                grads_mag[idx] = float("inf")

    smallest = np.argmin(grads_mag)
    if smallest == float("inf"): # if all are ignored tokens, return.
        return _instance, smallest

    # remove smallest
    inputs_before_smallest = text_field.tokens[0:smallest]
    inputs_after_smallest = text_field.tokens[smallest + 1:]
    text_field.tokens = inputs_before_smallest + inputs_after_smallest

    if "tags" in _instance:
        tag_field_before_smallest = tag_field.labels[0:smallest]
        tag_field_after_smallest = tag_field.labels[smallest + 1:]
        tag_field.labels = tag_field_before_smallest + tag_field_after_smallest
        tag_field.sequence_field = TextField(text_field, text_field._token_indexers)

    _instance.indexed = False
    return _instance, smallest

def get_ner_tags_and_mask(_current_instance: Instance,
                          _input_field_to_attack: str,
                          _ignore_tokens: List[str]):
    """
    Used for the NER task. Sets the num_ignore tokens, saves the original
    predicted tag and a 0/1 mask in the position of the tags
    """
    # Set num_ignore_tokens
    num_ignore_tokens = 0
    input_field = _current_instance[_input_field_to_attack]
    for token in input_field.tokens:
        if str(token) in _ignore_tokens:
            num_ignore_tokens += 1

    # save the original tags and a 0/1 mask where the tags are
    tag_mask = []
    original_tags = []
    tag_field: SequenceLabelField = _current_instance["tags"] # type: ignore
    for label in tag_field:
        if label != "O":
            tag_mask.append(1)
            original_tags.append(label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)
    return num_ignore_tokens, tag_mask, original_tags
