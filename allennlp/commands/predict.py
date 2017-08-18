import argparse
from contextlib import ExitStack
import json
from typing import Optional, IO

from allennlp.common.params import Params
from allennlp.service.predictors import Predictor

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Run the specified model against a JSON-lines input file.'''
    subparser = parser.add_parser(
            'predict', description=description, help='Use a trained model to make predictions.')
    subparser.add_argument('config_file', type=str, help='the configuration file that trained the model')
    subparser.add_argument('input_file', metavar='input-file', type=str, help='path to input file')
    subparser.add_argument('--output-file', type=str, help='path to output file')
    subparser.add_argument('--print', action='store_true', help='print results to string')

    subparser.set_defaults(func=predict)

    return subparser

def get_predictor(args: argparse.Namespace) -> Predictor:
    config = Params.from_file(args.config_file)
    predictor = Predictor.from_config(config)
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

    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(open(args.input_file, 'r'))  # type: ignore
        if args.output_file:
            output_file = stack.enter_context(open(args.output_file, 'w'))  # type: ignore
        else:
            output_file = None

        run(predictor, input_file, output_file, args.print)
