"""
The `evaluate_bias_mitigation` subcommand can be used to
compare a bias-mitigated trained model with a baseline
against an SNLI dataset following the format in [On Measuring
and Mitigating Biased Inferences of Word Embeddings]
(https://arxiv.org/pdf/1908.09369.pdf) and reports the
Net Neutral, Fraction Neutral, and Threshold:tau metrics.
"""

import argparse
import json
import logging
from typing import Any, Dict, Tuple
from overrides import overrides
import tempfile
import torch
import os

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.util import prepare_environment
from allennlp.fairness.bias_metrics import NaturalLanguageInference
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
            "baseline_archive_file", type=str, help="path to a baseline archived trained model"
        )

        subparser.add_argument(
            "input_file", type=str, help="path to the file containing the SNLI evaluation data"
        )

        subparser.add_argument("batch_size", type=int, help="batch size to use during evaluation")

        subparser.add_argument(
            "--bias-mitigated-output-file",
            type=str,
            help="optional path to write the metrics to as JSON",
        )

        subparser.add_argument(
            "--baseline-output-file",
            type=str,
            help="optional path to write the metrics to as JSON",
        )

        subparser.add_argument(
            "--bias-mitigated-predictions-file",
            type=str,
            help="optional path to write bias-mitigated predictions to as JSON",
        )

        subparser.add_argument(
            "--baseline-predictions-file",
            type=str,
            help="optional path to write baseline predictions to as JSON",
        )

        subparser.add_argument(
            "--predictions-diff-output-file",
            type=str,
            help="optional path to write diff of bias-mitigated and baseline predictions to as JSON",
        )

        subparser.add_argument(
            "--taus",
            type=float,
            nargs="+",
            default=[0.5, 0.7],
            help="tau parameters for Threshold metric",
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "--bias-mitigation-overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the bias mitigation experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--baseline-overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the baseline experiment configuration, e.g., "
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

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


class SNLIPredictionsDiff:
    def __init__(self):
        self.diff = []

    def __call__(self, bias_mitigated_labels, baseline_labels, original_tokens, tokenizer):
        """
        Returns label changes induced by bias mitigation and the corresponding sentence pairs.
        """
        for idx, label in enumerate(bias_mitigated_labels):
            if label != baseline_labels[idx]:
                self.diff.append(
                    {
                        "sentence_pair": tokenizer.convert_tokens_to_string(original_tokens[idx]),
                        "bias_mitigated_label": label,
                        "baseline_label": baseline_labels[idx],
                    }
                )

    def get_diff(self):
        return self.diff


# TODO: allow bias mitigation and baseline evaluations to run simultaneously on
# two different GPUs
def evaluate_from_args(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # Load from bias-mitigated archive
    bias_mitigated_archive = load_archive(
        args.bias_mitigated_archive_file,
        cuda_device=args.cuda_device,
        overrides=args.bias_mitigation_overrides,
    )
    bias_mitigated_config = bias_mitigated_archive.config
    prepare_environment(bias_mitigated_config)
    bias_mitigated_model = bias_mitigated_archive.model
    bias_mitigated_model.eval()

    # Load from baseline archive
    baseline_archive = load_archive(
        args.baseline_archive_file, cuda_device=args.cuda_device, overrides=args.baseline_overrides
    )
    baseline_config = baseline_archive.config
    prepare_environment(baseline_config)
    baseline_model = baseline_archive.model
    baseline_model.eval()

    # Load the evaluation data
    bias_mitigated_dataset_reader = bias_mitigated_archive.validation_dataset_reader
    baseline_dataset_reader = baseline_archive.validation_dataset_reader

    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    bias_mitigated_data_loader_params = bias_mitigated_config.pop("validation_data_loader", None)
    if bias_mitigated_data_loader_params is None:
        bias_mitigated_data_loader_params = bias_mitigated_config.pop("data_loader")
    # override batch sampler if exists
    if "batch_sampler" in bias_mitigated_data_loader_params:
        del bias_mitigated_data_loader_params["batch_sampler"]
    bias_mitigated_data_loader_params["batch_size"] = args.batch_size
    bias_mitigated_data_loader = DataLoader.from_params(
        params=bias_mitigated_data_loader_params,
        reader=bias_mitigated_dataset_reader,
        data_path=evaluation_data_path,
    )
    bias_mitigated_data_loader.index_with(bias_mitigated_model.vocab)

    baseline_data_loader_params = baseline_config.pop("validation_data_loader", None)
    if baseline_data_loader_params is None:
        baseline_data_loader_params = baseline_config.pop("data_loader")
    # override batch sampler if exists
    if "batch_sampler" in baseline_data_loader_params:
        del baseline_data_loader_params["batch_sampler"]
    baseline_data_loader_params["batch_size"] = args.batch_size
    baseline_data_loader = DataLoader.from_params(
        params=baseline_data_loader_params,
        reader=baseline_dataset_reader,
        data_path=evaluation_data_path,
    )
    baseline_data_loader.index_with(baseline_model.vocab)

    if args.bias_mitigated_predictions_file:
        bias_mitigated_filename = args.bias_mitigated_predictions_file
        bias_mitigated_file = os.open(bias_mitigated_filename, os.O_RDWR)
    else:
        bias_mitigated_file, bias_mitigated_filename = tempfile.mkstemp()
    bias_mitigated_output_metrics = evaluate(
        bias_mitigated_model,
        bias_mitigated_data_loader,
        args.cuda_device,
        predictions_output_file=bias_mitigated_filename,
    )

    if args.baseline_predictions_file:
        baseline_filename = args.baseline_predictions_file
        baseline_file = os.open(baseline_filename, os.O_RDWR)
    else:
        baseline_file, baseline_filename = tempfile.mkstemp()
    baseline_output_metrics = evaluate(
        baseline_model,
        baseline_data_loader,
        args.cuda_device,
        predictions_output_file=baseline_filename,
    )

    create_diff = hasattr(baseline_dataset_reader, "_tokenizer")
    if create_diff:
        diff_tool = SNLIPredictionsDiff()
    bias_mitigated_nli = NaturalLanguageInference(
        neutral_label=bias_mitigated_model.vocab.get_token_index("neutral", "labels"),
        taus=args.taus,
    )
    baseline_nli = NaturalLanguageInference(
        neutral_label=baseline_model.vocab.get_token_index("neutral", "labels"),
        taus=args.taus,
    )
    with open(bias_mitigated_file, "r") as bias_mitigated_fd, open(
        baseline_file, "r"
    ) as baseline_fd:
        for bias_mitigated_line, baseline_line in zip(bias_mitigated_fd, baseline_fd):
            bias_mitigated_predictions = json.loads(bias_mitigated_line)
            probs = torch.tensor(bias_mitigated_predictions["probs"])
            bias_mitigated_nli(probs)

            baseline_predictions = json.loads(baseline_line)
            probs = torch.tensor(baseline_predictions["probs"])
            baseline_nli(probs)

            if create_diff:
                diff_tool(
                    bias_mitigated_predictions["label"],
                    baseline_predictions["label"],
                    baseline_predictions["tokens"],
                    baseline_dataset_reader._tokenizer.tokenizer,  # type: ignore
                )

    bias_mitigated_metrics = {**bias_mitigated_output_metrics, **(bias_mitigated_nli.get_metric())}
    metrics_json = json.dumps(bias_mitigated_metrics, indent=2)
    if args.bias_mitigated_output_file:
        # write all metrics to output file
        # don't use dump_metrics() because want to log regardless
        with open(args.bias_mitigated_output_file, "w") as fd:
            fd.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    baseline_metrics = {**baseline_output_metrics, **(baseline_nli.get_metric())}
    metrics_json = json.dumps(baseline_metrics, indent=2)
    if args.baseline_output_file:
        # write all metrics to output file
        # don't use dump_metrics() because want to log regardless
        with open(args.baseline_output_file, "w") as fd:
            fd.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    if create_diff:
        diff = diff_tool.get_diff()
        diff_json = json.dumps(diff, indent=2)
        if args.predictions_diff_output_file:
            with open(args.predictions_diff_output_file, "w") as fd:
                fd.write(diff_json)
        logger.info("Predictions diff: %s", diff_json)

    logger.info("Finished evaluating.")

    return bias_mitigated_metrics, baseline_metrics
