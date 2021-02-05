"""
The `interpret` subcommand can be used to
run interpretation methods on a trained model
against a dataset and report any metrics involved.
"""

import argparse
import json
import logging
from typing import Any, Dict

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common import Params, Registrable, Lazy
from allennlp.common.util import prepare_environment
from allennlp.data import DataLoader, DatasetReader
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate, read_all_datasets

logger = logging.getLogger(__name__)


@Subcommand.register("interpret")
class Interpret(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Interpret the specified model on the specified dataset."""
        subparser = parser.add_parser(self.name, description=description, help="Train a model.")

        subparser.add_argument(
            "param_path", type=str, help="path to parameter file describing the model to be trained"
        )

        subparser.add_argument("archive_file", type=str, help="path to an archived trained model")

        subparser.add_argument(
            "input_file", type=str, help="path to the file containing the evaluation data"
        )

        subparser.add_argument(
            "--output-file", type=str, help="optional path to write the metrics to as JSON"
        )

        subparser.add_argument(
            "--predictions-output-file",
            type=str,
            help="optional path to write the predictions to as JSON lines",
        )

        subparser.add_argument(
            "--weights-file", type=str, help="a path that overrides which weights file to use"
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=interpret_from_args)

        return subparser


def interpret_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # TODO: finish modifying this
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(
        args.archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model
    # model.eval()

    # Load the interpreter config from a file
    params = Params.from_file(args.param_path, args.overrides)

    # Load the evaluation data
    validation_dataset_reader = archive.validation_dataset_reader
    overridden_validation_dataset_reader = DatasetReader.from_params(params=params)
    if overridden_validation_dataset_reader is not None:
        validation_dataset_reader = overridden_validation_dataset_reader

    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = validation_dataset_reader.read(evaluation_data_path)

    # if args.extend_vocab:
    #     logger.info("Vocabulary is being extended with test instances.")
    #     model.vocab.extend_from_instances(instances=instances)
    #     model.extend_embedder_vocab(embedding_sources)

    # instances.index_with(model.vocab)
    # data_loader_params = config.pop("validation_data_loader", None)
    # if data_loader_params is None:
    #     data_loader_params = config.pop("data_loader")
    # if args.batch_size:
    #     data_loader_params["batch_size"] = args.batch_size
    # data_loader = DataLoader.from_params(dataset=instances, params=data_loader_params)

    # metrics = evaluate(
    #     model,
    #     data_loader,
    #     args.cuda_device,
    #     args.batch_weight_key,
    #     output_file=args.output_file,
    #     predictions_output_file=args.predictions_output_file,
    # )

    logger.info("Finished evaluating.")

    return metrics
