"""
The ``predict`` subcommand allows you to make bulk JSON-to-JSON
predictions using a trained model and its :class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ allennlp predict --help
    usage: allennlp [command] predict [-h]
                                      [--output-file OUTPUT_FILE]
                                      [--batch-size BATCH_SIZE]
                                      [--silent]
                                      [--cuda-device CUDA_DEVICE]
                                      [-o OVERRIDES]
                                      [--include-package INCLUDE_PACKAGE]
                                      [--predictor PREDICTOR]
                                      archive_file input_file

    Run the specified model against a JSON-lines input file.

    positional arguments:
    archive_file          the archived model to make predictions with
    input_file            path to input file

    optional arguments:
    -h, --help            show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file
    --batch-size BATCH_SIZE
                            The batch size to use for processing
    --silent              do not print output to stdout
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a HOCON structure used to override the experiment
                            configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
    --predictor PREDICTOR
                            optionally specify a specific predictor to use
"""

import argparse
from contextlib import ExitStack
import sys
from typing import Optional, IO

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

class Predict(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')

        subparser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.set_defaults(func=_predict)

        return subparser

def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_file,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)

    return Predictor.from_archive(archive, args.predictor)

def _run(predictor: Predictor,
         input_file: IO,
         output_file: Optional[IO],
         batch_size: int,
         print_to_console: bool) -> None:

    def _run_predictor(batch_data):
        if len(batch_data) == 1:
            result = predictor.predict_json(batch_data[0])
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
        else:
            results = predictor.predict_batch_json(batch_data)

        for model_input, output in zip(batch_data, results):
            string_output = predictor.dump_line(output)
            if print_to_console:
                print("input: ", model_input)
                print("prediction: ", string_output)
            if output_file:
                output_file.write(string_output)

    batch_json_data = []
    for line in input_file:
        if not line.isspace():
            # Collect batch size amount of data.
            json_data = predictor.load_line(line)
            batch_json_data.append(json_data)
            if len(batch_json_data) == batch_size:
                _run_predictor(batch_json_data)
                batch_json_data = []

    # We might not have a dataset perfectly divisible by the batch size,
    # so tidy up the scraps.
    if batch_json_data:
        _run_predictor(batch_json_data)


def _predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)
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

        _run(predictor,
             input_file,
             output_file,
             args.batch_size,
             not args.silent)
