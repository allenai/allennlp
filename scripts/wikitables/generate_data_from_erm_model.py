#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position,protected-access
import sys
import os
import gzip
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util
from allennlp.models.archival import load_archive


def make_data(input_examples_file: str,
              tables_directory: str,
              archived_model_file: str,
              output_dir: str,
              num_logical_forms: int) -> None:
    reader = WikiTablesDatasetReader(tables_directory=tables_directory,
                                     keep_if_no_logical_forms=True,
                                     output_agendas=True)
    dataset = reader.read(input_examples_file)
    input_lines = []
    with open(input_examples_file) as input_file:
        input_lines = input_file.readlines()
    archive = load_archive(archived_model_file)
    model = archive.model
    model.training = False
    model._decoder_trainer._max_num_decoded_sequences = 100
    for instance, example_line in zip(dataset, input_lines):
        outputs = model.forward_on_instance(instance)
        world = instance.fields['world'].metadata
        parsed_info = util.parse_example_line(example_line)
        example_id = parsed_info["id"]
        target_list = parsed_info["target_values"]
        logical_forms = outputs["logical_form"]
        correct_logical_forms = []
        for logical_form in logical_forms:
            if world.evaluate_logical_form(logical_form, target_list):
                correct_logical_forms.append(logical_form)
                if len(correct_logical_forms) >= num_logical_forms:
                    break
        num_found = len(correct_logical_forms)
        print(f"{num_found} found for {example_id}")
        if num_found == 0:
            continue
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = gzip.open(os.path.join(output_dir, f"{example_id}.gz"), "wb")
        for logical_form in correct_logical_forms:
            logical_form_line = (logical_form + "\n").encode('utf-8')
            output_file.write(logical_form_line)
        output_file.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", type=str, help="Input file")
    argparser.add_argument("tables_directory", type=str, help="Tables directory")
    argparser.add_argument("archived_model", type=str, help="Archived model.tar.gz")
    argparser.add_argument("--output-dir", type=str, dest="output_dir", help="Output directory",
                           default="erm_output")
    argparser.add_argument("--num-logical-forms", type=int, dest="num_logical_forms",
                           help="Number of logical forms to output", default=10)
    args = argparser.parse_args()
    make_data(args.input, args.tables_directory, args.archived_model, args.output_dir,
              args.num_logical_forms)
