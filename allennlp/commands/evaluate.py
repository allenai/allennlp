"""
The `evaluate` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.
"""

import argparse
import json
import logging
from json import JSONDecodeError
from pathlib import Path
from os import PathLike
from typing import Union, Dict, Any, Optional
from copy import deepcopy

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.util import prepare_environment
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.evaluation import Evaluator

logger = logging.getLogger(__name__)


@Subcommand.register("evaluate")
class Evaluate(Subcommand):
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Evaluate the specified model + dataset(s)"""
        subparser = parser.add_parser(
            self.name, description=description, help="Evaluate the specified model + dataset(s)."
        )

        subparser.add_argument("archive_file", type=str, help="path to an archived trained model")

        subparser.add_argument(
            "input_file",
            type=str,
            help="path to the file containing the evaluation data (for multiple "
            "files, put between filenames e.g., input1.txt,input2.txt)",
        )

        subparser.add_argument(
            "--output-file",
            type=str,
            help="optional path to write the metrics to as JSON (for multiple "
            "files, put  between filenames e.g., output1.txt,output2.txt)",
        )

        subparser.add_argument(
            "--predictions-output-file",
            type=str,
            help="optional path to write the predictions to as JSON lines "
            "(for mutiple files, put  between filenames e.g., "
            "output1.jsonl,output2.jsonl)",
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
            help="Automatically create output names for each evaluation file. "
            "`NONE` will not automatically generate a file name for the "
            "neither the metrics nor the predictions. In this case you will"
            " need to pas in both `metrics_output_file` and `predictions_output_file`. "
            "`METRICS` will only automatically create a file name for the"
            " metrics file. `PREDS` will only automatically create a file"
            " name for the predictions outputs. `ALL` will create a "
            "filename for both the metrics and the predictions.",
            choices=["NONE", "METRICS", "PREDS", "ALL"],
        )

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return evaluate_from_archive(
        archive_file=args.archive_file,
        input_file=args.input_file,
        metrics_output_file=args.output_file,
        predictions_output_file=args.predictions_output_file,
        batch_size=args.batch_size,
        cmd_overrides=args.overrides,
        cuda_device=args.cuda_device,
        embedding_sources_mapping=args.embedding_sources_mapping,
        extend_vocab=args.extend_vocab,
        weights_file=args.weights_file,
        file_friendly_logging=args.file_friendly_logging,
        batch_weight_key=args.batch_weight_key,
        auto_names=args.auto_names,
    )


def evaluate_from_archive(
    archive_file: Union[str, PathLike],
    input_file: str,
    metrics_output_file: Optional[str] = None,
    predictions_output_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    cmd_overrides: Union[str, Dict[str, Any]] = "",
    cuda_device: int = -1,
    embedding_sources_mapping: str = None,
    extend_vocab: bool = False,
    weights_file: str = None,
    file_friendly_logging: bool = False,
    batch_weight_key: str = None,
    auto_names: str = "NONE",
) -> Dict[str, Any]:
    """

    # Parameters

    archive_file: `Union[str, PathLike]`
        Path to an archived trained model.

    input_file: `str`
        path to the file containing the evaluation data (for multiple files,
         put ":" between filenames e.g., input1.txt:input2.txt)

    metrics_output_file: `str`, optional (default=`None`)
        optional path to write the metrics to as JSON (for multiple files, put
         ":" between filenames e.g., output1.txt:output2.txt)

    predictions_output_file: `str`, optional (default=`None`)
        "optional path to write the predictions to (for multiple files, put ":"
         between filenames e.g., output1.jsonl:output2.jsonl)

    batch_size: `int`, optional (default=`None`)
        If non-empty, the batch size to use during evaluation.

    cmd_overrides: `str`, optional (default=`""`)
        a json(net) structure used to override the experiment configuration,
         e.g., '{\"iterator.batch_size\": 16}'.  Nested parameters can be
          specified either with nested dictionaries or with dot syntax.

    cuda_device: `int`, optional (default=`-1`)
        id of GPU to use (if any)

    embedding_sources_mapping: `str`, optional (default=`None`)
        a JSON dict defining mapping from embedding module path to embedding
        pretrained-file used during training. If not passed, and embedding
        needs to be extended, we will try to use the original file paths used
        during training. If they are not available we will use random vectors
        for embedding extension.

    extend_vocab: `bool`, optional (default=`False`)
        if specified, we will use the instances in your new dataset to extend
        your vocabulary. If pretrained-file was used to initialize embedding
        layers, you may also need to pass --embedding-sources-mapping.

    weights_file:`str`, optional (default=`None`)
        A path that overrides which weights file to use

    file_friendly_logging : `bool`, optional (default=`False`)
        If `True`, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.

    batch_weight_key: `str`, optional (default=`None`)
        If non-empty, name of metric used to weight the loss on a per-batch basis.

    auto_names:`str`, optional (default=`"NONE"`)
        Automatically create output names for each evaluation file.`NONE` will
        not automatically generate a file name for the neither the metrics nor
        the predictions. In this case you will need to pas in both
        `metrics_output_file` and `predictions_output_file`. `METRICS` will only
        automatically create a file name for the metrics file. `PREDS` will only
        automatically create a file name for the predictions outputs. `ALL`
        will create a filename for both the metrics and the predictions.

    # Returns

    all_metrics: `Dict[str, Any]`
        The metrics from every evaluation file passed.

    """
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
    evaluator_params["cuda_device"] = cuda_device
    evaluator = Evaluator.from_params(evaluator_params)

    # Load the evaluation data
    dataset_reader = archive.validation_dataset_reader

    # split files
    try:
        # Try reading it as a list of JSON objects first. Some readers require
        # that kind of input.
        evaluation_data_path_list = json.loads(f"[{input_file}]")
    except JSONDecodeError:
        evaluation_data_path_list = input_file.split(",")

    # TODO(gabeorlanski): Is it safe to always default to .outputs and .preds?
    # TODO(gabeorlanski): Add in way to save to specific output directory
    if metrics_output_file is not None:
        if auto_names == "METRICS" or auto_names == "ALL":
            logger.warning(
                f"Passed output_files will be ignored, auto_names is set to {auto_names}"
            )

            # Keep the path of the parent otherwise it will write to the CWD
            assert all(isinstance(p, str) for p in evaluation_data_path_list), (
                "When specifying JSON blobs as input, the output files must be explicitly named with "
                "--output-file."
            )
            output_file_list = [
                p.parent.joinpath(f"{p.stem}.outputs") for p in map(Path, evaluation_data_path_list)
            ]
        else:
            output_file_list = metrics_output_file.split(",")  # type: ignore
            assert len(output_file_list) == len(evaluation_data_path_list), (
                "The number of `metrics_output_file` paths must be equal to the number "
                "of datasets being evaluated."
            )
    if predictions_output_file is not None:
        if auto_names == "PREDS" or auto_names == "ALL":
            logger.warning(
                f"Passed predictions files will be ignored, auto_names is" f" set to {auto_names}"
            )

            # Keep the path of the parent otherwise it will write to the CWD
            assert all(isinstance(p, str) for p in evaluation_data_path_list), (
                "When specifying JSON blobs as input, the predictions output files must be explicitly named with "
                "--predictions-output-file."
            )
            predictions_output_file_list = [
                p.parent.joinpath(f"{p.stem}.preds") for p in map(Path, evaluation_data_path_list)
            ]
        else:
            predictions_output_file_list = predictions_output_file.split(",")  # type: ignore
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
    for index, evaluation_data_path in enumerate(evaluation_data_path_list):
        config = deepcopy(archive.config)

        # Get the eval file name so we can save each metric by file name in the
        # output dictionary.
        if isinstance(evaluation_data_path, str):
            eval_file_name = Path(evaluation_data_path).stem
        else:
            eval_file_name = str(index)

        if metrics_output_file is not None:
            # noinspection PyUnboundLocalVariable
            output_file_path = output_file_list[index]

        if predictions_output_file is not None:
            # noinspection PyUnboundLocalVariable
            predictions_output_file_path = predictions_output_file_list[index]

        logger.info("Reading evaluation data from %s", eval_file_name)
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
            metrics_output_file=output_file_path,
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
