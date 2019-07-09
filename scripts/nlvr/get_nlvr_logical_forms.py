#! /usr/bin/env python
import json
import argparse
from typing import Tuple, List
import os
import sys

# pylint: disable=wrong-import-position,invalid-name

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.common.util import JsonDict
from allennlp.semparse.domain_languages import NlvrLanguage
from allennlp.semparse.domain_languages.nlvr_language import Box
from allennlp.semparse import ActionSpaceWalker


def read_json_line(line: str) -> Tuple[str, str, List[JsonDict], List[str]]:
    data = json.loads(line)
    instance_id = data["identifier"]
    sentence = data["sentence"]
    if "worlds" in data:
        structured_reps = data["worlds"]
        label_strings = [label_str.lower() for label_str in data["labels"]]
    else:
        # We're reading ungrouped data.
        structured_reps = [data["structured_rep"]]
        label_strings = [data["label"].lower()]
    return instance_id, sentence, structured_reps, label_strings


def process_data(input_file: str,
                 output_file: str,
                 max_path_length: int,
                 max_num_logical_forms: int,
                 ignore_agenda: bool,
                 write_sequences: bool) -> None:
    """
    Reads an NLVR dataset and returns a JSON representation containing sentences, labels, correct and
    incorrect logical forms. The output will contain at most `max_num_logical_forms` logical forms
    each in both correct and incorrect lists. The output format is:
        ``[{"id": str, "label": str, "sentence": str, "correct": List[str], "incorrect": List[str]}]``
    """
    processed_data: JsonDict = []
    # We can instantiate the ``ActionSpaceWalker`` with any world because the action space is the
    # same for all the ``NlvrLanguage`` objects. It is just the execution that differs.
    walker = ActionSpaceWalker(NlvrLanguage({}), max_path_length=max_path_length)
    for line in open(input_file):
        instance_id, sentence, structured_reps, label_strings = read_json_line(line)
        worlds = []
        for structured_representation in structured_reps:
            boxes = set([Box(object_list, box_id) for box_id, object_list in enumerate(structured_representation)])
            worlds.append(NlvrLanguage(boxes))
        labels = [label_string == "true" for label_string in label_strings]
        correct_logical_forms = []
        incorrect_logical_forms = []
        if ignore_agenda:
            # Get 1000 shortest logical forms.
            logical_forms = walker.get_all_logical_forms(max_num_logical_forms=1000)
        else:
            # TODO (pradeep): Assuming all worlds give the same agenda.
            sentence_agenda = worlds[0].get_agenda_for_sentence(sentence)
            logical_forms = walker.get_logical_forms_with_agenda(sentence_agenda,
                                                                 max_num_logical_forms * 10)
        for logical_form in logical_forms:
            if all([world.execute(logical_form) == label for world, label in zip(worlds, labels)]):
                if len(correct_logical_forms) <= max_num_logical_forms:
                    correct_logical_forms.append(logical_form)
            else:
                if len(incorrect_logical_forms) <= max_num_logical_forms:
                    incorrect_logical_forms.append(logical_form)
            if len(correct_logical_forms) >= max_num_logical_forms \
               and len(incorrect_logical_forms) >= max_num_logical_forms:
                break
        if write_sequences:
            correct_sequences = [worlds[0].logical_form_to_action_sequence(logical_form) for logical_form in
                                 correct_logical_forms]
            incorrect_sequences = [worlds[0].logical_form_to_action_sequence(logical_form) for logical_form in
                                   incorrect_logical_forms]
            processed_data.append({"id": instance_id,
                                   "sentence": sentence,
                                   "correct_sequences": correct_sequences,
                                   "incorrect_sequences": incorrect_sequences,
                                   "worlds": structured_reps,
                                   "labels": label_strings})
        else:
            processed_data.append({"id": instance_id,
                                   "sentence": sentence,
                                   "correct_logical_forms": correct_logical_forms,
                                   "incorrect_logical_forms": incorrect_logical_forms,
                                   "worlds": structured_reps,
                                   "labels": label_strings})
    with open(output_file, "w") as outfile:
        for instance_processed_data in processed_data:
            json.dump(instance_processed_data, outfile)
            outfile.write('\n')
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="NLVR data file")
    parser.add_argument("output", type=str, help="Processed output")
    parser.add_argument("--max-path-length", type=int, dest="max_path_length",
                        help="Maximum path length for logical forms", default=12)
    parser.add_argument("--max-num-logical-forms", type=int, dest="max_num_logical_forms",
                        help="Maximum number of logical forms per denotation, per question",
                        default=20)
    parser.add_argument("--ignore-agenda", dest="ignore_agenda", help="Should we ignore the "
                        "agenda and use consistency as the only signal to get logical forms?",
                        action='store_true')
    parser.add_argument("--write-action-sequences", dest="write_sequences", help="If this "
                        "flag is set, action sequences instead of logical forms will be written "
                        "to the json file. This will avoid having to parse the logical forms again "
                        "in the NlvrDatasetReader.", action='store_true')
    args = parser.parse_args()
    process_data(args.input,
                 args.output,
                 args.max_path_length,
                 args.max_num_logical_forms,
                 args.ignore_agenda,
                 args.write_sequences)
