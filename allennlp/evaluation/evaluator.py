"""
Evaluator class for evaluating a model with a given dataset
"""
from collections import defaultdict
from typing import Union, List, Dict, Any, Tuple, Optional
from os import PathLike
from pathlib import Path
import torch
from itertools import groupby
import logging
import json
import numpy as np

from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, sanitize, int_to_device, END_SYMBOL, START_SYMBOL
from allennlp.data.fields import TensorField
from allennlp.nn import util as nn_util
from allennlp.common import Registrable, Params
from allennlp.models import Model
from allennlp.data import DataLoader, Vocabulary
from allennlp.evaluation.postprocessors.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class Evaluator(Registrable):
    """
    Evaluation class
    
    # Parameters
    
    cuda_device : `Union[int, torch.device]`, optional (default=`-1`)
        The cuda device to use for this evaluation.  The model is assumed to already be using this
        device; this parameter is only used for moving the input data to the correct device.
    """
    default_implementation = "simple"

    def __init__(
            self,
            batch_postprocessor: Postprocessor,
            cuda_device: Union[int, torch.device] = -1
    ):
        self.batch_human_serializer = batch_postprocessor
        self.cuda_device = cuda_device

    def __call__(
            self,
            model: Model,
            data_loader: DataLoader,
            batch_weight_key: str = None,
            output_file: Union[str, PathLike] = None,
            predictions_file: Union[str, PathLike] = None
    ):
        """
        # Parameters

        model : `Model`
            The model to evaluate
        data_loader : `DataLoader`
            The `DataLoader` that will iterate over the evaluation data (data loaders already contain
            their data).
        batch_weight_key : `str`, optional (default=`None`)
            If given, this is a key in the output dictionary for each batch that specifies how to weight
            the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
        metrics_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the final metrics to.
        predictions_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the predictions to.

        # Returns

        `Dict[str, Any]`
            The final metrics.
        """
        raise NotImplementedError("__call__")


@Evaluator.register("simple")
class SimpleEvaluator(Evaluator):
    def __init__(
            self,
            batch_postprocessor,
            cuda_device: Union[int, torch.device] = -1
    ):
        super(SimpleEvaluator, self).__init__(batch_postprocessor, cuda_device)

    def __call__(
            self,
            model: Model,
            data_loader: DataLoader,
            batch_weight_key: str = None,
            output_file: Union[str, PathLike] = None,
            predictions_file: Union[str, PathLike] = None
    ):
        """
        # Parameters

        model : `Model`
            The model to evaluate
        data_loader : `DataLoader`
            The `DataLoader` that will iterate over the evaluation data (data loaders already contain
            their data).
        batch_weight_key : `str`, optional (default=`None`)
            If given, this is a key in the output dictionary for each batch that specifies how to weight
            the loss for that batch.  If this is not given, we use a weight of 1 for every batch.
        metrics_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the final metrics to.
        predictions_output_file : `Union[str, PathLike]`, optional (default=`None`)
            Optional path to write the predictions to.

        # Returns

        `Dict[str, Any]`
            The final metrics.
        """
        check_for_gpu(self.cuda_device)
        data_loader.set_target_device(int_to_device(self.cuda_device))
        output_file = Path(output_file) if output_file is not None else None
        if predictions_file is not None:
            predictions_file = Path(predictions_file).open("w", encoding="utf-8")

        with torch.no_grad():
            model.eval()

            iterator = iter(data_loader)
            logger.info("Iterating over dataset")
            generator_tqdm = Tqdm.tqdm(iterator)
            # Number of batches in instances.
            batch_count = 0
            # Number of batches where the model produces a loss.
            loss_count = 0
            # Cumulative weighted loss
            total_loss = 0.0
            # Cumulative weight across all batches.
            total_weight = 0.0

            for batch in generator_tqdm:
                batch_count += 1
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output_dict = model(**batch)
                loss = output_dict.get("loss")

                metrics = model.get_metrics()

                if loss is not None:
                    loss_count += 1
                    if batch_weight_key:
                        weight = output_dict[batch_weight_key].item()
                    else:
                        weight = 1.0

                    total_weight += weight
                    total_loss += loss.item() * weight
                    # Report the average loss so far.
                    metrics["loss"] = total_loss / total_weight

                description = (
                        ", ".join(
                            [
                                "%s: %.2f" % (name, value)
                                for name, value in metrics.items()
                                if not name.startswith("_")
                            ]
                        )
                        + " ||"
                )
                generator_tqdm.set_description(description, refresh=False)

                # TODO(gabeorlanski): Add in postprocessing the batch for token
                #  metrics
                if predictions_file is not None:
                    predictions_file.write(
                        self.batch_human_serializer(
                            batch, output_dict, data_loader
                        ) + '\n'
                    )

            if predictions_file is not None:
                predictions_file.close()

            final_metrics = model.get_metrics(reset=True)
            if loss_count > 0:
                # Sanity check
                if loss_count != batch_count:
                    raise RuntimeError(
                        "The model you are trying to evaluate only sometimes produced a loss!"
                    )
                final_metrics["loss"] = total_loss / total_weight

            if output_file is not None:
                dump_metrics(str(output_file), final_metrics, log=True)

            return final_metrics

    def _to_params(self) -> Dict[str, Any]:
        return {
            "type"               : "simple",
            "cuda_device"        : self.cuda_device,
            "batch_postprocessor": self.batch_human_serializer.to_params()
        }
