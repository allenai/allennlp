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


@Attacker.register("input-reduction")
class InputReduction(Attacker):
    """
    Runs the input reduction method from [Pathologies of Neural Models Make Interpretations
    Difficult](https://arxiv.org/abs/1804.07781), which removes as many words as possible from
    the input without changing the model's prediction.

    The functions on this class handle a special case for NER by looking for a field called "tags"
    This check is brittle, i.e., the code could break if the name of this field has changed, or if
    a non-NER model has a field called "tags".

    Registered as an `Attacker` with name "input-reduction".
    """

    def __init__(self, predictor: Predictor, beam_size: int = 3) -> None:
        super().__init__(predictor)
        self.beam_size = beam_size

    def attack_from_json(
        self,
        inputs: JsonDict = None,
        input_field_to_attack: str = "tokens",
        grad_input_field: str = "grad_input_1",
        ignore_tokens: List[str] = None,
        target: JsonDict = None,
    ):
        if target is not None:
            raise ValueError("Input reduction does not implement targeted attacks")
        ignore_tokens = ["@@NULL@@"] if ignore_tokens is None else ignore_tokens
        original_instances = self.predictor.json_to_labeled_instances(inputs)
        original_text_field: TextField = original_instances[0][  # type: ignore
            input_field_to_attack
        ]
        original_tokens = deepcopy(original_text_field.tokens)
        final_tokens = []
        for instance in original_instances:
            final_tokens.append(
                self._attack_instance(
                    inputs, instance, input_field_to_attack, grad_input_field, ignore_tokens
                )
            )
        return sanitize({"final": final_tokens, "original": original_tokens})

    def _attack_instance(
        self,
        inputs: JsonDict,
        instance: Instance,
        input_field_to_attack: str,
        grad_input_field: str,
        ignore_tokens: List[str],
    ):
        # Save fields that must be checked for equality
        fields_to_compare = utils.get_fields_to_compare(inputs, instance, input_field_to_attack)

        # Set num_ignore_tokens, which tells input reduction when to stop
        # We keep at least one token for input reduction on classification/entailment/etc.
        if "tags" not in instance:
            num_ignore_tokens = 1
            tag_mask = None

        # Set num_ignore_tokens for NER and build token mask
        else:
            num_ignore_tokens, tag_mask, original_tags = _get_ner_tags_and_mask(
                instance, input_field_to_attack, ignore_tokens
            )

        text_field: TextField = instance[input_field_to_attack]  # type: ignore
        current_tokens = deepcopy(text_field.tokens)
        candidates = [(instance, -1, tag_mask)]
        # keep removing tokens until prediction is about to change
        while len(current_tokens) > num_ignore_tokens and candidates:
            # sort current candidates by smallest length (we want to remove as many tokens as possible)
            def get_length(input_instance: Instance):
                input_text_field: TextField = input_instance[input_field_to_attack]  # type: ignore
                return len(input_text_field.tokens)

            candidates = heapq.nsmallest(self.beam_size, candidates, key=lambda x: get_length(x[0]))

            beam_candidates = deepcopy(candidates)
            candidates = []
            for beam_instance, smallest_idx, tag_mask in beam_candidates:
                # get gradients and predictions
                beam_tag_mask = deepcopy(tag_mask)
                grads, outputs = self.predictor.get_gradients([beam_instance])

                for output in outputs:
                    if isinstance(outputs[output], torch.Tensor):
                        outputs[output] = outputs[output].detach().cpu().numpy().squeeze().squeeze()
                    elif isinstance(outputs[output], list):
                        outputs[output] = outputs[output][0]

                # Check if any fields have changed, if so, next beam
                if "tags" not in instance:
                    # relabel beam_instance since last iteration removed an input token
                    beam_instance = self.predictor.predictions_to_labeled_instances(
                        beam_instance, outputs
                    )[0]
                    if utils.instance_has_changed(beam_instance, fields_to_compare):
                        continue

                # special case for sentence tagging (we have tested NER)
                else:
                    # remove the mask where you remove the input token from.
                    if smallest_idx != -1:  # Don't delete on the very first iteration
                        del beam_tag_mask[smallest_idx]
                    cur_tags = [
                        outputs["tags"][x] for x in range(len(outputs["tags"])) if beam_tag_mask[x]
                    ]
                    if cur_tags != original_tags:
                        continue

                # remove a token from the input
                text_field: TextField = beam_instance[input_field_to_attack]  # type: ignore
                current_tokens = deepcopy(text_field.tokens)
                reduced_instances_and_smallest = _remove_one_token(
                    beam_instance,
                    input_field_to_attack,
                    grads[grad_input_field][0],
                    ignore_tokens,
                    self.beam_size,
                    beam_tag_mask,
                )
                candidates.extend(reduced_instances_and_smallest)
        return current_tokens


def _remove_one_token(
    instance: Instance,
    input_field_to_attack: str,
    grads: np.ndarray,
    ignore_tokens: List[str],
    beam_size: int,
    tag_mask: List[int],
) -> List[Tuple[Instance, int, List[int]]]:
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
    reduced_instances_and_smallest: List[Tuple[Instance, int, List[int]]] = []
    for _ in range(beam_size):
        # copy instance and edit later
        copied_instance = deepcopy(instance)
        copied_text_field: TextField = copied_instance[input_field_to_attack]  # type: ignore

        # find smallest
        smallest = np.argmin(grads_mag)
        if grads_mag[smallest] == float("inf"):  # if all are ignored tokens, return.
            break
        grads_mag[smallest] = float("inf")  # so the other beams don't use this token

        # remove smallest
        inputs_before_smallest = copied_text_field.tokens[0:smallest]
        inputs_after_smallest = copied_text_field.tokens[smallest + 1 :]
        copied_text_field.tokens = inputs_before_smallest + inputs_after_smallest

        if "tags" in instance:
            tag_field: SequenceLabelField = copied_instance["tags"]  # type: ignore
            tag_field_before_smallest = tag_field.labels[0:smallest]
            tag_field_after_smallest = tag_field.labels[smallest + 1 :]
            tag_field.labels = tag_field_before_smallest + tag_field_after_smallest  # type: ignore
            tag_field.sequence_field = copied_text_field

        copied_instance.indexed = False
        reduced_instances_and_smallest.append((copied_instance, smallest, tag_mask))

    return reduced_instances_and_smallest


def _get_ner_tags_and_mask(
    instance: Instance, input_field_to_attack: str, ignore_tokens: List[str]
):
    """
    Used for the NER task. Sets the num_ignore tokens, saves the original predicted tag and a 0/1
    mask in the position of the tags
    """
    # Set num_ignore_tokens
    num_ignore_tokens = 0
    input_field: TextField = instance[input_field_to_attack]  # type: ignore
    for token in input_field.tokens:
        if str(token) in ignore_tokens:
            num_ignore_tokens += 1

    # save the original tags and a 0/1 mask where the tags are
    tag_mask = []
    original_tags = []
    tag_field: SequenceLabelField = instance["tags"]  # type: ignore
    for label in tag_field.labels:
        if label != "O":
            tag_mask.append(1)
            original_tags.append(label)
            num_ignore_tokens += 1
        else:
            tag_mask.append(0)
    return num_ignore_tokens, tag_mask, original_tags
