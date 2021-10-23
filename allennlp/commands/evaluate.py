"""
The `evaluate` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.
"""

import argparse
import json
import logging
from pathlib import Path
from os import PathLike
from typing import Union, Dict, Any, Optional, List
from copy import deepcopy

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.util import prepare_environment
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.evaluation import Evaluator

logger = logging.getLogger(__name__)


@Subcommand.register("evaluate")
class Evaluate(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Evaluate the specified model + dataset(s)"""
        subparser = parser.add_parser(
            self.name, description=description, help="Evaluate the specified model + dataset(s)."
        )

        subparser.add_argument("archive_file", type=str, help="path to an archived trained model")

        subparser.add_argument(
            "input_file",
            type=str,
            help=(
                "path to the file containing the evaluation data"
                ' (for mutiple files, put ":" between filenames e.g., input1.txt:input2.txt)'
            ),
        )

        subparser.add_argument(
            "--output-file",
            type=str,
            help=(
                "optional path to write the metrics to as JSON"
                ' (for mutiple files, put ":" between filenames e.g., output1.txt:output2.txt)'
            ),
        )

        subparser.add_argument(
            "--predictions-output-file",
            type=str,
            help=(
                "optional path to write the predictions to as JSON lines"
                ' (for mutiple files, put ":" between filenames e.g., output1.jsonl:output2.jsonl)'
            ),
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
            "--batch-size", type=int, help="If non-empty, the batch size to use during evaluation."
        )

        subparser.add_argument(
            "--batch-weight-key",
            type=str,
            default="",
            help="If non-empty, name of metric used to weight the loss on a per-batch basis.",
        )

        subparser.add_argument(
            "--extend-vocab",
            action="store_true",
            default=False,
            help="if specified, we will use the instances in your new dataset to "
                 "extend your vocabulary. If pretrained-file was used to initialize "
                 "embedding layers, you may also need to pass --embedding-sources-mapping.",
        )

        subparser.add_argument(
            "--embedding-sources-mapping",
            type=str,
            default="",
            help="a JSON dict defining mapping from embedding module path to embedding "
                 "pretrained-file used during training. If not passed, and embedding needs to be "
                 "extended, we will try to use the original file paths used during training. If "
                 "they are not available we will use random vectors for embedding extension.",
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.add_argument(
            "--auto-names",
            default="NONE",
            help="Automatically create output names for each evaluation file.",
            choices=["NONE", "METRICS", "PREDS", "ALL"]
        )

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return evaluate_from_archive(
        archive_file=args.archive_file,
        input_file=args.input_file,
        output_file=args.output_file,
        predictions_output_file=args.predictions_output_file,
        batch_size=args.batch_size,
        cmd_overrides=args.overrides,
        cuda_device=args.cuda_device,
        embedding_sources_mapping=args.embedding_sources_mapping,
        extend_vocab=args.extend_vocab,
        weights_file=args.weights_file,
        file_friendly_logging=args.file_friendly_logging,
        batch_weight_key=args.batch_weight_key,
        auto_names=args.auto_names
    )


def evaluate_from_archive(
        archive_file: Union[str, PathLike],
        input_file: str,
        output_file: Optional[str] = None,
        predictions_output_file: Optional[str] = None,
        batch_size: Optional[int] = None,
        cmd_overrides: Union[str, Dict[str, Any]] = "",
        cuda_device: int = -1,
        embedding_sources_mapping: str = None,
        extend_vocab: bool = False,
        weights_file: str = None,
        file_friendly_logging: bool = False,
        batch_weight_key: str = None,
        auto_names: str = "None",
):
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(
        archive_file,
        weights_file=weights_file,
        cuda_device=cuda_device,
        overrides=cmd_overrides,
    )
    config = deepcopy(archive.config)
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluator from the config key `Evaluator`
    evaluator_params = config.pop("evaluation", {})
    evaluator_params['cuda_device'] = cuda_device
    evaluator = Evaluator.from_params(evaluator_params)

    # Load the evaluation data
    dataset_reader = archive.validation_dataset_reader

    # split files
    evaluation_data_path_list = input_file.split(":")

    # TODO(gabeorlanski): Is it safe to always default to .outputs and .preds?
    # TODO(gabeorlanski): Add in way to save to specific output directory
    if output_file is not None:
        if auto_names == "METRICS" or auto_names == "ALL":
            logger.warning(f"Passed output_files will be ignored, auto_names is"
                           f" set to {auto_names}")

            # Keep the path of the parent otherwise it will write to the CWD
            output_file_list = [
                p.parent.joinpath(f"{p.stem}.outputs")
                for p in map(Path, evaluation_data_path_list)
            ]
        else:
            output_file_list = output_file.split(":")
            assert len(output_file_list) == len(
                evaluation_data_path_list
            ), "The number of `output_file` paths must be equal to the number of datasets being evaluated."
    if predictions_output_file is not None:
        if auto_names == "PREDS" or auto_names == "ALL":
            logger.warning(f"Passed predictions files will be ignored, auto_names is"
                           f" set to {auto_names}")

            # Keep the path of the parent otherwise it will write to the CWD
            predictions_output_file_list = [
                p.parent.joinpath(f"{p.stem}.preds")
                for p in map(Path, evaluation_data_path_list)
            ]
        else:
            predictions_output_file_list = predictions_output_file.split(":")
            assert len(predictions_output_file_list) == len(evaluation_data_path_list), (
                    "The number of `predictions_output_file` paths must be equal"
                    + "to the number of datasets being evaluated. "
            )

    # output file
    output_file_path = None
    predictions_output_file_path = None

    # embedding sources
    if extend_vocab:
        logger.info("Vocabulary is being extended with embedding sources.")
        embedding_sources = (
            json.loads(embedding_sources_mapping) if embedding_sources_mapping else {}
        )

    all_metrics = {}
    for index in range(len(evaluation_data_path_list)):
        config = deepcopy(archive.config)
        evaluation_data_path = evaluation_data_path_list[index]

        # Get the eval file name so we can save each metric by file name in the
        # output dictionary.
        eval_file_name = Path(evaluation_data_path).stem

        if output_file is not None:
            # noinspection PyUnboundLocalVariable
            output_file_path = output_file_list[index]

        if predictions_output_file is not None:
            # noinspection PyUnboundLocalVariable
            predictions_output_file_path = predictions_output_file_list[index]

        logger.info("Reading evaluation data from %s", evaluation_data_path)
        data_loader_params = config.get("validation_data_loader", None)
        if data_loader_params is None:
            data_loader_params = config.get("data_loader")
        if batch_size:
            data_loader_params["batch_size"] = batch_size
        data_loader = DataLoader.from_params(
            params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
        )

        if extend_vocab:
            logger.info("Vocabulary is being extended with test instances.")
            model.vocab.extend_from_instances(instances=data_loader.iter_instances())
            # noinspection PyUnboundLocalVariable
            model.extend_embedder_vocab(embedding_sources)

        data_loader.index_with(model.vocab)

        metrics = evaluator(
            model,
            data_loader,
            batch_weight_key,
            output_file=output_file_path,
            predictions_output_file=predictions_output_file_path,
        )

        # Add the metric prefixed by the file it came from.
        for name, value in metrics.items():
            if len(evaluation_data_path_list) > 1:
                key = f"{eval_file_name}_"
            else:
                key = ""
            all_metrics[f"{key}{name}"] = value

    logger.info("Finished evaluating.")

    return all_metrics
