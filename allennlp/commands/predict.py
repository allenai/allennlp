"""
The ``predict`` subcommand allows you to make bulk JSON-to-JSON
predictions using a trained model and its :class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ python -m allennlp.run predict --help
    usage: python -m allennlp.run [command] predict [-h]
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
from typing import Optional, IO, Dict

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'srl': 'semantic-role-labeling',
        'decomposable_attention': 'textual-entailment',
        'bidaf': 'machine-comprehension',
        'simple_tagger': 'sentence-tagger',
        'crf_tagger': 'sentence-tagger',
        'coref': 'coreference-resolution'
}


class Predict(Subcommand):
    def __init__(self, predictor_overrides: Dict[str, str] = {}) -> None:
        # pylint: disable=dangerous-default-value
        self.predictors = {**DEFAULT_PREDICTORS, **predictor_overrides}

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')

        subparser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')
        batch_size.add_argument('--batch_size', type=int, help=argparse.SUPPRESS)

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
        cuda_device.add_argument('--cuda_device', type=int, help=argparse.SUPPRESS)

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.set_defaults(func=_predict(self.predictors))

        return subparser

def _get_predictor(args: argparse.Namespace, predictors: Dict[str, str]) -> Predictor:
    archive = load_archive(args.archive_file, cuda_device=args.cuda_device, overrides=args.overrides)

    if args.predictor:
        # Predictor explicitly specified, so use it
        return Predictor.from_archive(archive, args.predictor)

    # Otherwise, use the mapping
    model_type = archive.config.get("model").get("type")
    if model_type not in predictors:
        raise ConfigurationError("no known predictor for model type {}".format(model_type))
    return Predictor.from_archive(archive, predictors[model_type])

def _run(predictor: Predictor,
         input_file: IO,
         output_file: Optional[IO],
         batch_size: int,
         print_to_console: bool,
         cuda_device: int) -> None:

    def _run_predictor(batch_data):
        if len(batch_data) == 1:
            result = predictor.predict_json(batch_data[0], cuda_device)
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
        else:
            results = predictor.predict_batch_json(batch_data, cuda_device)

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


def _predict(predictors: Dict[str, str]):
    def predict_inner(args: argparse.Namespace) -> None:
        # Import any additional modules needed (to register custom classes)
        for package_name in args.include_package:
            import_submodules(package_name)

        predictor = _get_predictor(args, predictors)
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

            _run(predictor, input_file, output_file, args.batch_size, not args.silent, args.cuda_device)

    return predict_inner
