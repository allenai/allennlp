#! /usr/bin/env python
import json
import argparse
from typing import Tuple, List
import os
import sys

# pylint: disable=wrong-import-position,invalid-name

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.common.util import JsonDict
from allennlp.semparse.worlds import NlvrWorld
from allennlp.semparse import ActionSpaceWalker


def read_json_line(line: str) -> Tuple[str, str, List[NlvrWorld], List[bool]]:
    data = json.loads(line)
    instance_id = data["identifier"]
    sentence = data["sentence"]
    if "worlds" in data:
        worlds = [NlvrWorld(json_dict) for json_dict in data["worlds"]]
        labels = [label_str.lower() == "true" for label_str in data["labels"]]
    else:
        # We're reading ungrouped data.
        worlds = [NlvrWorld(data["structured_rep"])]
        labels = [data["label"].lower() == "true"]
    return instance_id, sentence, worlds, labels


def process_data(input_file: str,
                 output_file: str,
                 max_path_length: int,
                 max_num_logical_forms: int) -> None:
    """
    Reads an NLVR dataset and returns a JSON representation containing sentences, labels, correct and
    incorrect logical forms. The output will contain at most `max_num_logical_forms` logical forms
    each in both correct and incorrect lists. The output format is:
        ``[{"id": str, "label": str, "sentence": str, "correct": List[str], "incorrect": List[str]}]``
    """
    processed_data: JsonDict = []
    # We can instantiate the ``ActionSpaceWalker`` with any world because the action space is the
    # same for all the ``NlvrWorlds``. It is just the execution that differs.
    walker = ActionSpaceWalker(NlvrWorld({}), max_path_length=max_path_length)
    for line in open(input_file):
        instance_id, sentence, worlds, labels = read_json_line(line)
        correct_logical_forms = []
        incorrect_logical_forms = []
        # TODO (pradeep): Assuming all worlds give the same agenda.
        sentence_agenda = worlds[0].get_agenda_for_sentence(sentence, add_paths_to_agenda=False)
        if sentence_agenda:
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
        processed_data.append({"id": instance_id,
                               "sentence": sentence,
                               "correct": correct_logical_forms,
                               "incorrect": incorrect_logical_forms})
    outfile = open(output_file, "w")
    json.dump(processed_data, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="NLVR data file")
    parser.add_argument("output", type=str, help="Processed output")
    parser.add_argument("--max_path_length", type=int,
                        help="Maximum path length for logical forms", default=12)
    parser.add_argument("--max_num_logical_forms", type=int,
                        help="Maximum number of logical forms per denotation, per question",
                        default=20)
    args = parser.parse_args()
    process_data(args.input, args.output, args.max_path_length, args.max_num_logical_forms)
