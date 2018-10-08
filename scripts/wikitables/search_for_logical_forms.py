#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position
import sys
import os
import argparse
from pyparsing import OneOrMore, nestedExpr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.semparse import ActionSpaceWalker
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.worlds import WikiTablesVariableFreeWorld
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.data.tokenizers import WordTokenizer
from allennlp.tools import wikitables_evaluator as evaluator


# TODO (pradeep): Use a DatasetReader (when available) instead of directly reading from the examples
# file.


def search(tables_directory: str,
           input_examples_file: str,
           output_file: str,
           max_path_length: int,
           max_num_logical_forms: int,
           use_agenda: bool) -> None:
    lisp_parser = OneOrMore(nestedExpr())
    data = lisp_parser.parseFile(input_examples_file)
    tokenizer = WordTokenizer()
    with open(output_file, "w") as output_file_pointer:
        for instance_data in data:
            utterance = instance_data[2][1]
            question_id = instance_data[1][1]
            if utterance.startswith('"') and utterance.endswith('"'):
                utterance = utterance[1:-1]
            # For example: csv/200-csv/47.csv -> tagged/200-tagged/47.tagged
            table_file = instance_data[3][1][2].replace("csv", "tagged")
            # The first element is the word "list"
            sempre_target_list = instance_data[4][1].asList()[1:]
            # Each item in the list is of the form ["description", "item"]. Removing "description"s.
            target_list = []
            for list_item in sempre_target_list:
                target = list_item[1]
                if target.startswith('"') and target.endswith('"'):
                    target = list_item[1][1:-1]
                # pylint: disable=protected-access
                target_list.append(TableQuestionContext._normalize_string(target))
            target_value_list = evaluator.to_value_list(target_list)
            tokenized_question = tokenizer.tokenize(utterance)
            table_file = f"{tables_directory}/{table_file}"
            context = TableQuestionContext.read_from_file(table_file, tokenized_question)
            world = WikiTablesVariableFreeWorld(context)
            walker = ActionSpaceWalker(world, max_path_length=max_path_length)
            correct_logical_forms = []
            print(f"{question_id} {utterance}", file=output_file_pointer)
            if use_agenda:
                agenda = world.get_agenda()
                print(f"Agenda: {agenda}", file=output_file_pointer)
                all_logical_forms = walker.get_logical_forms_with_agenda(agenda=agenda,
                                                                         max_num_logical_forms=10000)
            else:
                all_logical_forms = walker.get_all_logical_forms(max_num_logical_forms=10000)
            for logical_form in all_logical_forms:
                try:
                    denotation = world.execute(logical_form)
                except ExecutionError:
                    print(f"Failed to execute: {logical_form}", file=sys.stderr)
                    continue
                if isinstance(denotation, list):
                    denotation_list = [str(denotation_item) for denotation_item in denotation]
                else:
                    # For numbers and dates
                    denotation_list = [str(denotation)]
                denotation_value_list = evaluator.to_value_list(denotation_list)
                if evaluator.check_denotation(target_value_list, denotation_value_list):
                    correct_logical_forms.append(logical_form)
            if not correct_logical_forms:
                print("NO LOGICAL FORMS FOUND!", file=output_file_pointer)
            for logical_form in correct_logical_forms[:max_num_logical_forms]:
                print(logical_form, file=output_file_pointer)
            print(file=output_file_pointer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_directory", type=str, help="Location of the 'tagged' directory in the"
                        "WikiTableQuestions dataset")
    parser.add_argument("data_file", type=str, help="Path to the *.examples file")
    # TODO (pradeep): We may eventually want to produce gzipped files like DPD output instead of
    # writing all logical forms in one file.
    parser.add_argument("output_file", type=str, help="Path to the output file")
    parser.add_argument("--max-path-length", type=int, dest="max_path_length", default=10,
                        help="Max length to which we will search exhaustively")
    parser.add_argument("--max-num-logical-forms", type=int, dest="max_num_logical_forms",
                        default=100, help="Maximum number of logical forms returned")
    parser.add_argument("--use-agenda", dest="use_agenda", action="store_true")
    args = parser.parse_args()
    search(args.table_directory, args.data_file, args.output_file, args.max_path_length,
           args.max_num_logical_forms, args.use_agenda)
