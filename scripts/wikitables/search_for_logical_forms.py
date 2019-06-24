#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position
import sys
import os
import argparse
import gzip
import logging
import math
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.common.util import JsonDict
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util as wikitables_util
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.domain_languages import WikiTablesLanguage
from allennlp.semparse import ActionSpaceWalker

def search(tables_directory: str,
           data: JsonDict,
           output_path: str,
           max_path_length: int,
           max_num_logical_forms: int,
           use_agenda: bool,
           output_separate_files: bool,
           conservative_agenda: bool) -> None:
    print(f"Starting search with {len(data)} instances", file=sys.stderr)
    language_logger = logging.getLogger('allennlp.semparse.domain_languages.wikitables_language')
    language_logger.setLevel(logging.ERROR)
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
        world = WikiTablesLanguage(context)
        walker = ActionSpaceWalker(world, max_path_length=max_path_length)
        correct_logical_forms = []
        if use_agenda:
            agenda = world.get_agenda(conservative=conservative_agenda)
            allow_partial_match = not conservative_agenda
            all_logical_forms = walker.get_logical_forms_with_agenda(agenda=agenda,
                                                                     max_num_logical_forms=10000,
                                                                     allow_partial_match=allow_partial_match)
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
    parser.add_argument("--use-agenda", dest="use_agenda", action="store_true",
                        help="Use agenda while searching for logical forms")
    parser.add_argument("--conservative", action="store_true",
                        help="Get conservative agenda, and select logical forms with complete match.")
    parser.add_argument("--output-separate-files", dest="output_separate_files",
                        action="store_true", help="""If set, the script will output gzipped
                        files, one per example. You may want to do this if you;re making data to
                        train a parser.""")
    parser.add_argument("--num-splits", dest="num_splits", type=int, default=0,
                        help="Number of splits to make of the data, to run as many processes (default 0)")
    args = parser.parse_args()
    input_data = [wikitables_util.parse_example_line(example_line) for example_line in
                  open(args.data_file)]
    if args.num_splits == 0 or len(input_data) <= args.num_splits or not args.output_separate_files:
        search(args.table_directory, input_data, args.output_path, args.max_path_length,
               args.max_num_logical_forms, args.use_agenda, args.output_separate_files,
               args.conservative)
    else:
        chunk_size = math.ceil(len(input_data)/args.num_splits)
        start_index = 0
        for i in range(args.num_splits):
            if i == args.num_splits - 1:
                data_split = input_data[start_index:]
            else:
                data_split = input_data[start_index:start_index + chunk_size]
            start_index += chunk_size
            process = Process(target=search, args=(args.table_directory, data_split,
                                                   args.output_path, args.max_path_length,
                                                   args.max_num_logical_forms, args.use_agenda,
                                                   args.output_separate_files,
                                                   args.conservative))
            print(f"Starting process {i}", file=sys.stderr)
            process.start()
