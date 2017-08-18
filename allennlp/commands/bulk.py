import argparse
from contextlib import ExitStack
import json
import sys
from typing import Optional, IO

from allennlp.service.predictors import Predictor, load_predictors

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Run the specified model against a JSON-lines input file.'''
    subparser = parser.add_parser(
            'bulk', description=description, help='Run a model in bulk.')
    subparser.add_argument('model', type=str, help='the name of the model to run')
    subparser.add_argument('input_file', metavar='input-file', type=str, help='path to input file')
    subparser.add_argument('--output-file', type=str, help='path to output file')
    subparser.add_argument('--print', action='store_true', help='print results to string')

    subparser.set_defaults(func=bulk)

    return subparser

def get_model(args: argparse.Namespace) -> Optional[Predictor]:
    # TODO(joelgrus): use the args to instantiate the model
    models = load_predictors()
    model_name = args.model
    return models.get(model_name)

def run(predictor: Predictor, input_file: IO, output_file: Optional[IO], print_to_console: bool) -> None:
    for line in input_file:
        data = json.loads(line)
        result = predictor.predict_json(data)
        output = json.dumps(result)

        if print_to_console:
            print(output)
        if output_file:
            output_file.write(output + "\n")

def bulk(args: argparse.Namespace) -> None:
    model = get_model(args)
    if model is None:
        print("unknown model:", args.model)
        sys.exit(-1)

    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(open(args.input_file, 'r'))  # type: ignore
        if args.output_file:
            output_file = stack.enter_context(open(args.output_file, 'w'))  # type: ignore
        else:
            output_file = None

        run(model, input_file, output_file, args.print)
