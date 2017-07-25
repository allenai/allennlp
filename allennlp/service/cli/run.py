import argparse
from contextlib import ExitStack
import json
import sys

from allennlp.service.servable import Servable, ServableCollection

parser = argparse.ArgumentParser(description="Run a model")  # pylint: disable=invalid-name
parser.add_argument('model', type=str, help='the name of the model to run')
parser.add_argument('input_file', metavar='input-file', type=str, help='path to input file')
parser.add_argument('--output-file', type=str, help='path to output file')
parser.add_argument('--print', action='store_true', help='print results to string')

def run(servable: Servable, input_fn: str, output_fn: str, print_to_console: bool) -> None:
    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(open(input_fn, 'r'))  # type: ignore
        if output_fn:
            output_file = stack.enter_context(open(output_fn, 'w'))  # type: ignore

        for line in input_file:
            data = json.loads(line)
            result = servable.predict_json(data)
            output = json.dumps(result)

            if print_to_console:
                print(output)
            if output_file:
                output_file.write(output + "\n")


def main():
    args = parser.parse_args()  # pylint: disable=invalid-name
    # TODO: make this configurable
    servables = ServableCollection.default() # pylint: disable=invalid-name
    model = servables.get(args.model)
    if model is None:
        print("unknown model:", args.model)
        sys.exit(-1)

    run(model, args.input_file, args.output_file, args.print)

if __name__ == "__main__":
    main()
