#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.data.dataset_readers import NlvrDatasetReader
from allennlp.models import NlvrCoverageSemanticParser
from allennlp.models.archival import load_archive
from allennlp.semparse.worlds import NlvrWorld


def make_data(input_file: str,
              output_file: str,
              archived_model_file: str,
              max_num_decoded_sequences: int) -> None:
    reader = NlvrDatasetReader(output_agendas=True)
    model = load_archive(archived_model_file).model
    if not isinstance(model, NlvrCoverageSemanticParser):
        model_type = type(model)
        raise RuntimeError(f"Expected an archived NlvrCoverageSemanticParser, but found {model_type} instead")
    # Tweaking the decoder trainer to coerce the it to generate a k-best list. Setting k to 100
    # here, so that we can filter out the inconsistent ones later.
    model._decoder_trainer._max_num_decoded_sequences = 100  # pylint: disable=protected-access
    num_outputs = 0
    num_sentences = 0
    with open(output_file, "w") as outfile:
        for line in open(input_file):
            num_sentences += 1
            input_data = json.loads(line)
            sentence = input_data["sentence"]
            structured_representations = input_data["worlds"]
            labels = input_data["labels"]
            instance = reader.text_to_instance(sentence, structured_representations)
            outputs = model.forward_on_instance(instance)
            action_strings = outputs["best_action_strings"]
            logical_forms = outputs["logical_form"]
            correct_sequences = []
            # Checking for consistency
            worlds = [NlvrWorld(structure) for structure in structured_representations]
            for sequence, logical_form in zip(action_strings, logical_forms):
                denotations = [world.execute(logical_form) for world in worlds]
                denotations_are_correct = [label.lower() == str(denotation).lower()
                                           for label, denotation in zip(labels, denotations)]
                if all(denotations_are_correct):
                    correct_sequences.append(sequence)
            correct_sequences = correct_sequences[:max_num_decoded_sequences]
            if not correct_sequences:
                continue
            output_data = {
                    "id": input_data["identifier"],
                    "sentence": sentence,
                    "correct_sequences": correct_sequences,
                    "worlds": structured_representations,
                    "labels": input_data["labels"]
            }
            json.dump(output_data, outfile)
            outfile.write('\n')
            num_outputs += 1
        outfile.close()
    sys.stderr.write(f"{num_outputs} out of {num_sentences} sentences have outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input data file")
    parser.add_argument("output", type=str, help="Output data file")
    parser.add_argument("archived_model", type=str, help="Path to archived model.tar.gz to use for decoding")
    parser.add_argument("--max-num-sequences", type=int, dest="max_num_sequences",
                        help="Maximum number of sequences per instance to output", default=20)
    args = parser.parse_args()
    make_data(args.input, args.output, args.archived_model, args.max_num_sequences)
