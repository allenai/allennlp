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
                            a JSON structure used to override the experiment
                            configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
    --predictor PREDICTOR
                            optionally specify a specific predictor to use
"""
from typing import List
import argparse
from contextlib import ExitStack
import sys

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

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
        subparser.add_argument('--dataset-reader',
                               type=str,
                               default=None,
                               help='A dataset reader to use to load instances from input_file')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

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



class _Predict:

    def __init__(self, predictor, input_file, output_file, batch_size, print_to_console, has_dataset_reader):

        self._predictor = predictor
        self._input_file = input_file
        self._output_file = output_file
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        if has_dataset_reader:
            self._dataset_reader = predictor._dataset_reader
        else:
            self._dataset_reader = None

    def _predict_json_lines(self, batch_data: List[JsonDict]):
        if len(batch_data) == 1:
            result = self._predictor.predict_json(batch_data[0])
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            yield self._predictor.dump_line(result)
        else:
            results = self._predictor.predict_batch_json(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _predict_on_instances(self, batch_data: List[Instance]):
        if len(batch_data) == 1:
            result = self._predictor.predict_instance(batch_data[0])
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            yield self._predictor.dump_line(result)
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _maybe_print_to_console_and_file(self, prediction: JsonDict, model_input: JsonDict = None) -> None:
        if self._print_to_console:
            if model_input:
                print("input: ", model_input)
            print("prediction: ", prediction)
        if self._output_file is not None:
            self._output_file.write(prediction)

    def _get_json_data(self):
        for line in self._input_file:
            if not line.isspace():
                yield self._predictor.load_line(line)
    def _get_instance_data(self):
        if self._dataset_reader is None:
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        else:
            yield from self._dataset_reader.read(self._input_file)

    def run(self):
        has_reader = self._dataset_reader is not None
        generator = self._get_instance_data() if has_reader else self._get_json_data()
        predict_function = self._predict_on_instances if has_reader else self._predict_json_lines

        for batch in lazy_groups_of(generator, self._batch_size):
            for model_input, result in zip(batch, predict_function(batch)):
                input_to_print = model_input if not has_reader else None
                self._maybe_print_to_console_and_file(result,
                                                      model_input=input_to_print)

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

        util = _Predict(predictor, input_file, output_file, args.batch_size, not args.silent, args.dataset_reader)

        util.run()
