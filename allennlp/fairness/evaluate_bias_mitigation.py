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
            "baseline_archive_file", type=str, help="path to a baseline archived trained model"
        )

        subparser.add_argument(
            "input_file", type=str, help="path to the file containing the SNLI evaluation data"
        )

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
            "--predictions-diff-output-file",
            type=str,
            help="optional path to write diff of bias-mitigated and baseline predictions to as JSON lines",
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


def _add(accumulator_dict, append_dict):
    """
    Adds list pairs in append_dict to accumulator_dict,
    concatenating list values for duplicate keys.
    """
    for k, v in append_dict.items():
        if isinstance(v, list):
            if k not in accumulator_dict:
                accumulator_dict[k] = []
            accumulator_dict[k] += v


def compute_metrics(probs, model, taus):
    """
    Computes the following metrics:

    1. Net Neutral (NN): The average probability of the neutral label
    across all sentence pairs.

    2. Fraction Neutral (FN): The fraction of sentence pairs predicted neutral.

    3. Threshold:tau (T:tau): A parameterized measure that reports the fraction
    of examples whose probability of neutral is above tau

    """
    metrics = {}
    neutral_label = model.vocab.get_token_index("neutral", "labels")

    metrics["net_neutral"] = probs[..., neutral_label].mean().item()
    metrics["fraction_neutral"] = (probs.argmax(dim=-1) == neutral_label).float().mean().item()
    for tau in taus:
        metrics["threshold_{}".format(tau)] = (
            (probs[..., neutral_label] > tau).float().mean().item()
        )

    return metrics


def compute_predictions_diff(bias_mitigated_labels, baseline_labels, tokens, baseline_tokenizer):
    """
    Returns label changes induced by bias mitigation and the corresponding sentence pairs.
    """
    diff = []
    for idx, label in enumerate(bias_mitigated_labels):
        if label != baseline_labels[idx]:
            diff.append(
                {
                    "sentence_pair": baseline_tokenizer.convert_tokens_to_string(tokens[idx]),
                    "bias_mitigated_label": label,
                    "baseline_label": baseline_labels[idx],
                }
            )
    return diff


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
    if args.batch_size:
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
    if args.batch_size:
        baseline_data_loader_params["batch_size"] = args.batch_size
    baseline_data_loader = DataLoader.from_params(
        params=baseline_data_loader_params,
        reader=baseline_dataset_reader,
        data_path=evaluation_data_path,
    )
    baseline_data_loader.index_with(baseline_model.vocab)

    bias_mitigated_file, bias_mitigated_filename = tempfile.mkstemp()
    bias_mitigated_output_metrics = evaluate(
        bias_mitigated_model,
        bias_mitigated_data_loader,
        args.cuda_device,
        predictions_output_file=bias_mitigated_filename,
    )

    bias_mitigated_predictions: Dict[str, Any] = {}
    with open(bias_mitigated_file, "r") as fd:
        for line in fd:
            _add(bias_mitigated_predictions, json.loads(line))

    probs = torch.tensor(bias_mitigated_predictions["probs"], device=args.cuda_device)
    bias_mitigated_metrics = compute_metrics(probs, bias_mitigated_model, args.taus)
    metrics_json = json.dumps({**bias_mitigated_output_metrics, **bias_mitigated_metrics}, indent=2)
    if args.bias_mitigated_output_file:
        # write all metrics to output file
        # don't use dump_metrics() because want to log regardless
        with open(args.bias_mitigated_output_file, "w") as fd:
            fd.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    baseline_file, baseline_filename = tempfile.mkstemp()
    baseline_output_metrics = evaluate(
        baseline_model,
        baseline_data_loader,
        args.cuda_device,
        predictions_output_file=baseline_filename,
    )

    baseline_predictions: Dict[str, Any] = {}
    with open(baseline_file, "r") as fd:
        for line in fd:
            _add(baseline_predictions, json.loads(line))

    probs = torch.tensor(baseline_predictions["probs"], device=args.cuda_device)
    baseline_metrics = compute_metrics(probs, baseline_model, args.taus)
    metrics_json = json.dumps({**baseline_output_metrics, **baseline_metrics}, indent=2)
    if args.baseline_output_file:
        # write all metrics to output file
        # don't use dump_metrics() because want to log regardless
        with open(args.baseline_output_file, "w") as fd:
            fd.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    if hasattr(baseline_dataset_reader, "_tokenizer"):
        diff = compute_predictions_diff(
            bias_mitigated_predictions["label"],
            baseline_predictions["label"],
            baseline_predictions["tokens"],
            baseline_dataset_reader._tokenizer.tokenizer,  # type: ignore
        )
        diff_json = json.dumps(diff, indent=2)
        if args.predictions_diff_output_file:
            with open(args.predictions_diff_output_file, "w") as fd:
                fd.write(diff_json)
        logger.info("Predictions diff: %s", diff_json)

    logger.info("Finished evaluating.")

    return bias_mitigated_metrics, baseline_metrics
