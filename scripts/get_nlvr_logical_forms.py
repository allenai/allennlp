#! /usr/bin/env python

import json
import argparse
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from allennlp.common.util import JsonDict
from allennlp.data.semparse.worlds import NlvrWorld
from allennlp.data.semparse import ActionSpaceWalker


def read_json_line(line: str) -> Tuple[str, str, NlvrWorld, bool]:
    data = json.loads(line)
    instance_id = data["identifier"]
    sentence = data["sentence"]
    world = NlvrWorld(data["structured_rep"])
    label = data["label"].lower() == "true"
    return instance_id, sentence, world, label


def process_data(input_file: str,
                 output_file: str,
                 max_path_length: int,
                 max_num_logical_forms: int) -> None:
    """
    Reads an NLVR dataset and returns a JSON representation containing sentences, correct and
    incorrect logical forms. The format is:
        ``[{"id": str, "sentence": str, "correct": List[str], "incorrect": List[str]}]``
    """
    processed_data: JsonDict = []
    # We can instantiate the ``ActionSpaceWalker`` with any world because the action space is the
    # same for all the ``NlvrWorlds``. It is just the execution that differs.
    walker = ActionSpaceWalker(NlvrWorld({}), max_path_length=max_path_length)
    for line in open(input_file):
        instance_id, sentence, world, label = read_json_line(line)
        sentence_agenda = world.get_agenda_for_sentence(sentence, add_paths_to_agenda=False)
        logical_forms = walker.get_logical_forms_with_agenda(sentence_agenda,
                                                             max_num_logical_forms * 10)
        correct_logical_forms = []
        incorrect_logical_forms = []
        for logical_form in logical_forms:
            if world.execute(logical_form) == label:
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
