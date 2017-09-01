"""
The ``predict`` subcommand allows you to make bulk JSON-to-JSON
predictions using a trained model and its :class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ python -m allennlp.run predict --help
    usage: run [command] predict [-h] [--output-file OUTPUT_FILE] [--print]
                                archive_file input-file

    Run the specified model against a JSON-lines input file.

    positional arguments:
    archive_file          the archived model to make predictions with
    input-file            path to input file

    optional arguments:
    -h, --help            show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file
    --print               print results to stdout
"""

import argparse
from contextlib import ExitStack
import json
import sys
from typing import Optional, IO

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Run the specified model against a JSON-lines input file.'''
    subparser = parser.add_parser(
            'predict', description=description, help='Use a trained model to make predictions.')
    subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
    subparser.add_argument('input_file', metavar='input-file', type=argparse.FileType('r'),
                           help='path to input file')
    subparser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')
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

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(args.input_file)  # type: ignore
        if args.output_file:
            output_file = stack.enter_context(args.output_file)  # type: ignore

        run(predictor, input_file, output_file, not args.silent)
