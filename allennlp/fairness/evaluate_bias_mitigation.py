"""
The `evaluate_bias_mitigation` subcommand can be used to
compare a bias-mitigated trained model with a baseline
against an SNLI dataset following the format in [On Measuring
and Mitigating Biased Inferences of Word Embeddings]
(https://arxiv.org/pdf/1908.09369.pdf) and reports the
Net Neutral, Fraction Neutral, and Threshold metrics.
"""

import argparse
import json
import logging
from typing import Any, Dict
from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.util import prepare_environment
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate

logger = logging.getLogger(__name__)


@Subcommand.register("evaluate-bias-mitigation")
class EvaluateBiasMitigation(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Evaluate bias mitigation"""
        subparser = parser.add_parser(
            self.name, description=description, help="Evaluate bias mitigation."
        )

        subparser.add_argument(
            "bias_mitigated_archive_file",
            type=str,
            help="path to a bias-mitigated archived trained model",
        )

        subparser.add_argument(
            "archive_file", type=str, help="path to a baseline archived trained model"
        )

        subparser.add_argument(
            "input_file", type=str, help="path to the file containing the evaluation data"
        )

        subparser.add_argument(
            "--output-file", type=str, help="optional path to write the metrics to as JSON"
        )

        subparser.add_argument(
            "--bias-mitigated-predictions-output-file",
            type=str,
            help="optional path to write bias-mitigated predictions to as JSON lines",
        )

        subparser.add_argument(
            "--baseline-predictions-output-file",
            type=str,
            help="optional path to write baseline predictions to as JSON lines",
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "--batch-size", type=int, help="If non-empty, the batch size to use during evaluation."
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # Load from bias-mitigated archive
    archive = load_archive(
        args.bias_mitigated_archive_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load from baseline archive
    archive = load_archive(
        args.bias_mitigated_archive_file,
        weights_file=args.weights_file,
        cuda_device=args.cuda_device,
        overrides=args.overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    dataset_reader = archive.validation_dataset_reader

    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    data_loader_params = config.pop("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.pop("data_loader")
    if args.batch_size:
        data_loader_params["batch_size"] = args.batch_size
    data_loader = DataLoader.from_params(
        params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
    )

    embedding_sources = (
        json.loads(args.embedding_sources_mapping) if args.embedding_sources_mapping else {}
    )

    if args.extend_vocab:
        logger.info("Vocabulary is being extended with test instances.")
        model.vocab.extend_from_instances(instances=data_loader.iter_instances())
        model.extend_embedder_vocab(embedding_sources)

    data_loader.index_with(model.vocab)

    metrics = evaluate(
        model,
        data_loader,
        args.cuda_device,
        args.batch_weight_key,
        output_file=args.output_file,
        predictions_output_file=args.predictions_output_file,
    )

    logger.info("Finished evaluating.")

    return metrics