#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position
import sys
import os
import argparse
import gzip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.semparse import ActionSpaceWalker
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.worlds import WikiTablesVariableFreeWorld
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util as wikitables_util


def search(tables_directory: str,
           input_examples_file: str,
           output_path: str,
           max_path_length: int,
           max_num_logical_forms: int,
           use_agenda: bool,
           output_separate_files: bool) -> None:
    data = [wikitables_util.parse_example_line(example_line) for example_line in
            open(input_examples_file)]
    tokenizer = WordTokenizer()
    if output_separate_files and not os.path.exists(output_path):
        os.makedirs(output_path)
    if not output_separate_files:
        output_file_pointer = open(output_path, "w")
    for instance_data in data:
        utterance = instance_data["question"]
        question_id = instance_data["id"]
        if utterance.startswith('"') and utterance.endswith('"'):
            utterance = utterance[1:-1]
        # For example: csv/200-csv/47.csv -> tagged/200-tagged/47.tagged
        table_file = instance_data["table_filename"].replace("csv", "tagged")
        target_list = instance_data["target_values"]
        tokenized_question = tokenizer.tokenize(utterance)
        table_file = f"{tables_directory}/{table_file}"
        context = TableQuestionContext.read_from_file(table_file, tokenized_question)
        world = WikiTablesVariableFreeWorld(context)
        walker = ActionSpaceWalker(world, max_path_length=max_path_length)
        correct_logical_forms = []
        if use_agenda:
            agenda = world.get_agenda()
            all_logical_forms = walker.get_logical_forms_with_agenda(agenda=agenda,
                                                                     max_num_logical_forms=10000)
        else:
            all_logical_forms = walker.get_all_logical_forms(max_num_logical_forms=10000)
        for logical_form in all_logical_forms:
            if world.evaluate_logical_form(logical_form, target_list):
                correct_logical_forms.append(logical_form)
        if output_separate_files and correct_logical_forms:
            with gzip.open(f"{output_path}/{question_id}.gz", "wt") as output_file_pointer:
                for logical_form in correct_logical_forms:
                    print(logical_form, file=output_file_pointer)
        elif not output_separate_files:
            print(f"{question_id} {utterance}", file=output_file_pointer)
            if use_agenda:
                print(f"Agenda: {agenda}", file=output_file_pointer)
            if not correct_logical_forms:
                print("NO LOGICAL FORMS FOUND!", file=output_file_pointer)
            for logical_form in correct_logical_forms[:max_num_logical_forms]:
                print(logical_form, file=output_file_pointer)
            print(file=output_file_pointer)
    if not output_separate_files:
        output_file_pointer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_directory", type=str, help="Location of the 'tagged' directory in the"
                        "WikiTableQuestions dataset")
    parser.add_argument("data_file", type=str, help="Path to the *.examples file")
    parser.add_argument("output_path", type=str, help="""Path to the output directory if
                        'output_separate_files' is set, or to the output file if not.""")
    parser.add_argument("--max-path-length", type=int, dest="max_path_length", default=10,
                        help="Max length to which we will search exhaustively")
    parser.add_argument("--max-num-logical-forms", type=int, dest="max_num_logical_forms",
                        default=100, help="Maximum number of logical forms returned")
    parser.add_argument("--use-agenda", dest="use_agenda", action="store_true")
    parser.add_argument("--output-separate-files", dest="output_separate_files",
                        action="store_true", help="""If set, the script will output gzipped
                        files, one per example. You may want to do this if you're making data to
                        train a parser.""")
    args = parser.parse_args()
    search(args.table_directory, args.data_file, args.output_path, args.max_path_length,
           args.max_num_logical_forms, args.use_agenda, args.output_separate_files)
