import argparse
from contextlib import ExitStack
import json
from typing import Optional, IO

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Run the specified model against a JSON-lines input file.'''
    subparser = parser.add_parser(
            'predict', description=description, help='Use a trained model to make predictions.')
    subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    subparser.add_argument('input_file', metavar='input-file', type=str, help='path to input file')
    subparser.add_argument('--output-file', type=str, help='path to output file')
    subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

    subparser.set_defaults(func=predict)

    return subparser

def get_predictor(args: argparse.Namespace) -> Predictor:
    archive = load_archive(args.archive_file)
    predictor = Predictor.from_archive(archive)
    return predictor

def run(predictor: Predictor, input_file: IO, output_file: Optional[IO], print_to_console: bool) -> None:
    for line in input_file:
        data = json.loads(line)
        result = predictor.predict_json(data)
        output = json.dumps(result)

        if print_to_console:
            print(output)
        if output_file:
            output_file.write(output + "\n")

def predict(args: argparse.Namespace) -> None:
    predictor = get_predictor(args)
    output_file = None

    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(args.input_file)  # type: ignore
        if args.output_file:
            output_file = stack.enter_context(args.output_file)  # type: ignore

        run(predictor, input_file, output_file, not args.silent)
