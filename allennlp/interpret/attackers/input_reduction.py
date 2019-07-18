from copy import deepcopy
from typing import List, Tuple
import heapq

import numpy as np
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.interpret.attackers import utils
from allennlp.interpret.attackers.attacker import Attacker
from allennlp.predictors import Predictor


@Attacker.register('input-reduction')
class InputReduction(Attacker):
    """
    Runs the input reduction method from `Pathologies of Neural Models Make Interpretations
    Difficult <https://arxiv.org/abs/1804.07781>`_, which removes as many words as possible from
    the input without changing the model's prediction.

    The functions on this class handle a special case for NER by looking for a field called "tags"
    This check is brittle, i.e., the code could break if the name of this field has changed, or if
    a non-NER model has a field called "tags".
    """
    def __init__(self, predictor: Predictor) -> None:
        super().__init__(predictor)
        self.beam_size = 3

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
                num_ignore_tokens, tag_mask, original_tags = _get_ner_tags_and_mask(current_instance,
                                                                                    input_field_to_attack,
                                                                                    ignore_tokens)

            current_text_field: TextField = current_instance[input_field_to_attack]  # type: ignore
            current_tokens = deepcopy(current_text_field.tokens)
            # new_
            current_candidates = [(current_instance, -1)]
            # keep removing tokens until prediction is about to change
            while len(current_tokens) > num_ignore_tokens and current_candidates:
                # sort current candidates by smallest length (we want to remove as many tokens as possible)
                def get_length(input_instance: Instance):
                    input_text_field: TextField = input_instance[input_field_to_attack]  # type: ignore
                    return len(input_text_field.tokens)
                current_candidates = heapq.nsmallest(self.beam_size,
                                                     current_candidates,
                                                     key=lambda x:get_length(x[0]))

                beam_candidates = deepcopy(current_candidates)
                current_candidates = []
                for beam_instance, smallest_idx in beam_candidates:
                    # get gradients and predictions
                    grads, outputs = self.predictor.get_gradients([beam_instance])

                    for output in outputs:
                        if isinstance(outputs[output], torch.Tensor):
                            outputs[output] = outputs[output].detach().cpu().numpy().squeeze().squeeze()
                        elif isinstance(outputs[output], list):
                            outputs[output] = outputs[output][0]

                    # Check if any fields have changed, if so, next beam
                    if "tags" not in current_instance:
                        # relabel beam_instance since last iteration removed an input token
                        beam_instance = self.predictor.predictions_to_labeled_instances(beam_instance, outputs)[0]
                        if any(beam_instance[field] != fields_to_compare[field] for field in fields_to_compare):
                            continue

                    # special case for sentence tagging (we have tested NER)
                    else:
                        # remove the mask where you deleted from.
                        # Don't delete on the very first iteration, or if another beam already deleted
                        # the mask for you
                        if len(tag_mask) > len(outputs["tags"]):
                            del tag_mask[smallest_idx]
                        cur_tags = [outputs["tags"][x] for x in range(len(outputs["tags"])) if tag_mask[x]]
                        if cur_tags != original_tags:
                            continue

                    # remove a token from the input
                    current_text_field: TextField = beam_instance[input_field_to_attack]  # type: ignore
                    current_tokens = deepcopy(current_text_field.tokens)
                    reduced_instances_and_smallest = _remove_one_token(beam_instance,
                                                                       input_field_to_attack,
                                                                       grads[grad_input_field],
                                                                       ignore_tokens,
                                                                       self.beam_size)
                    current_candidates.extend(reduced_instances_and_smallest)

            final_tokens.append(current_tokens)
        return sanitize({"final": final_tokens, "original": original_tokens})


def _remove_one_token(instance: Instance,
                      input_field_to_attack: str,
                      grads: np.ndarray,
                      ignore_tokens: List[str],
                      beam_size: int = 1) -> List[Tuple[Instance, int]]:
    """
    Finds the token with the smallest gradient and removes it.
    """
    # Compute L2 norm of all grads.
    grads_mag = [np.sqrt(grad.dot(grad)) for grad in grads]

    # Skip all ignore_tokens by setting grad to infinity
    text_field: TextField = instance[input_field_to_attack]  # type: ignore
    for token_idx, token in enumerate(text_field.tokens):
        if token in ignore_tokens:
            grads_mag[token_idx] = float("inf")

    # For NER, skip all tokens that are not in outside
    if "tags" in instance:
        tag_field: SequenceLabelField = instance["tags"]  # type: ignore
        labels: List[str] = tag_field.labels  # type: ignore
        for idx, label in enumerate(labels):
            if label != "O":
                grads_mag[idx] = float("inf")
    reduced_instances_and_smallest: List[Tuple[Instance, int]] = []
    for _ in range(beam_size):
        # copy instance and edit later
        copied_instance = deepcopy(instance)
        copied_text_field: TextField = copied_instance[input_field_to_attack] # type: ignore

        # find smallest
        smallest = np.argmin(grads_mag)
        if grads_mag[smallest] == float("inf"):  # if all are ignored tokens, return.
            break
        grads_mag[smallest] = float("inf") # so the other beams don't use this token

        # remove smallest
        inputs_before_smallest = copied_text_field.tokens[0:smallest]
        inputs_after_smallest = copied_text_field.tokens[smallest + 1:]
        copied_text_field.tokens = inputs_before_smallest + inputs_after_smallest

        if "tags" in instance:
            tag_field: SequenceLabelField = copied_instance["tags"]  # type: ignore
            tag_field_before_smallest = tag_field.labels[0:smallest]
            tag_field_after_smallest = tag_field.labels[smallest + 1:]
            tag_field.labels = tag_field_before_smallest + tag_field_after_smallest  # type: ignore
            tag_field.sequence_field = copied_text_field

        copied_instance.indexed = False
        reduced_instances_and_smallest.append((copied_instance, smallest))

    return reduced_instances_and_smallest

def _get_ner_tags_and_mask(current_instance: Instance,
                           input_field_to_attack: str,
                           ignore_tokens: List[str]):
    """
    Used for the NER task. Sets the num_ignore tokens, saves the original predicted tag and a 0/1
    mask in the position of the tags
    """
    # Set num_ignore_tokens
    num_ignore_tokens = 0
    input_field: TextField = current_instance[input_field_to_attack]  # type: ignore
    for token in input_field.tokens:
        if str(token) in ignore_tokens:
            num_ignore_tokens += 1

    # save the original tags and a 0/1 mask where the tags are
    tag_mask = []
    original_tags = []
    tag_field: SequenceLabelField = current_instance["tags"]  # type: ignore
    for label in tag_field.labels:
        if label != "O":
            tag_mask.append(1)
            original_tags.append(label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)
    return num_ignore_tokens, tag_mask, original_tags
