import argparse
from contextlib import ExitStack
import json
import sys

from allennlp.service.models import models
from allennlp.service.models.types import Model

parser = argparse.ArgumentParser(description="Run a model")  # pylint: disable=invalid-name
parser.add_argument('model', type=str, help='the name of the model to run')
parser.add_argument('input_file', metavar='input-file', type=str, help='path to input file')
parser.add_argument('--output-file', type=str, help='path to output file')
parser.add_argument('--print', action='store_true', help='print results to string')


def run_model(model: Model, input_fn: str, output_fn: str, print_to_console: bool) -> None:
    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(open(args.input_file, 'r'))
        if args.output_file:
            output_file = stack.enter_context(open(args.output_file, 'w'))

        for line in input_file:
            data = json.loads(line)
            result = model(data)
            output = json.dumps(result)

            if args.print:
                print(output)
            if args.output_file:
                output_file.write(output + "\n")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.model not in models:
        print("unknown model:", args.model)
        sys.exit(-1)

    run_model(models[args.model](), args.input_file, args.output_file, args.print)
