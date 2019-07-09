from typing import List
import numpy as np
import torch
from allennlp.interpret.attackers import Attacker
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Attacker.register('input-reduction')
class InputReduction(Attacker):
    """
    Runs the input reduction method from `Pathologies of Neural Models Make Interpretations
    Difficult` https://arxiv.org/abs/1804.07781, which removes as many words as possible
    from the input _without_ changing the model's prediction.
    """
    def attack_from_json(self, inputs: JsonDict,
                         target_field: str, gradient_index: str,
                         ignore_tokens: List[str] = ["@@NULL@@"]):
        original_instances = self.predictor.inputs_to_labeled_instances(inputs)
        final_tokens = []
        fields_to_check = {}
        for current_instances in original_instances:
            current_instances = [current_instances]
            original_tokens = [x for x in current_instances[0][target_field].tokens]

            # Save fields that must be checked for equality
            test_instances = self.predictor.inputs_to_labeled_instances(inputs)
            for key in current_instances[0].fields.keys():
                if key not in inputs.keys() and key != target_field:
                    fields_to_check[key] = test_instances[0][key]

            # Set num_ignore_tokens, which tells input reduction when to stop
            # Keep at least one token for classification/entailment/etc.
            if "tags" not in current_instances[0]:
                num_ignore_tokens = 1

            # Set num_ignore_tokens for NER and build token mask
            else:
                num_ignore_tokens, tag_mask, original_tags = \
                    get_ner_tags_and_mask(current_instances, target_field, ignore_tokens)
            current_tokens = current_instances[0][target_field].tokens
            smallest_idx = -1
            # keep removing tokens
            while len(current_instances[0][target_field]) >= num_ignore_tokens:
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
                current_tokens = list(current_instances[0][target_field].tokens)
                current_instances, smallest_idx = \
                    remove_one_token(grads[gradient_index], current_instances, target_field, ignore_tokens)

            final_tokens.append(current_tokens)
        return sanitize({"final": final_tokens, "original": original_tokens})

def remove_one_token(grads: np.ndarray,
                     instances: List[Instance],
                     target_field: str,
                     ignore_tokens: List[str] = ["@@NULL@@"]) -> List[Instance]:
    """
    Finds the token with the smallest gradient and removes it.
    """
    # Compute L2 norm of all grads.
    grads_mag = [np.sqrt(grad.dot(grad)) for grad in grads]

    # Skip all ignore_tokens by setting grad to infinity
    for tok_idx, tok in enumerate(instances[0][target_field].tokens):
        if tok in ignore_tokens:
            grads_mag[tok_idx] = float("inf")
    # For NER, skip all tokens that are not in outside
    if "tags" in instances[0]:
        for idx, label in enumerate(instances[0]["tags"].__dict__["field_list"]):
            if label.label != "O":
                grads_mag[idx] = float("inf")

    smallest = np.argmin(grads_mag)
    if smallest == float("inf"): # if all are ignored tokens, return.
        return instances

    # remove smallest
    instances[0][target_field].tokens = \
    instances[0][target_field].tokens[0:smallest] +  instances[0][target_field].tokens[smallest + 1:]
    if "tags" in instances[0]:
        instances[0]["tags"].__dict__["field_list"] = \
            instances[0]["tags"].__dict__["field_list"][0:smallest] + \
            instances[0]["tags"].__dict__["field_list"][smallest + 1:]

    instances[0].indexed = False
    return instances, smallest

def get_ner_tags_and_mask(current_instances: List[Instance],
                          target_field: str,
                          ignore_tokens: List[str] = ["@@NULL@@"]):
    """
    Used for the NER task. Sets the num_ignore tokens, saves the original
     predicted tag and a 0/1 mask in the position of the tags
    """
    # Set num_ignore_tokens
    num_ignore_tokens = 0
    for token in current_instances[0][target_field].tokens:
        if str(token) in ignore_tokens:
            num_ignore_tokens += 1

    # save the original tags and a 0/1 mask where the tags are
    tag_mask = []
    original_tags = []
    for label in current_instances[0]["tags"].__dict__["field_list"]:
        if label.label != "O":
            tag_mask.append(1)
            original_tags.append(label.label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)