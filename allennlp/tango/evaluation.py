"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

import dataclasses
from typing import Dict, Any, List, Optional

import torch

from allennlp.common import Lazy, Tqdm
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import sanitize
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from allennlp.tango.dataloader import TangoDataLoader, BatchSizeDataLoader
from allennlp.tango.dataset import DatasetDict
from allennlp.tango.format import JsonFormat, Format
from allennlp.tango.step import Step


@Step.register("evaluation")
class EvaluationStep(Step):
    """This step evaluates a given model on a given dataset."""

    DETERMINISTIC = True
    VERSION = "002"
    FORMAT: Format = JsonFormat(compress="gz")

    @dataclasses.dataclass
    class EvaluationResult:
        metrics: Dict[str, Any]
        predictions: List[Dict[str, Any]]

    def run(  # type: ignore
        self,
        model: Model,
        dataset: DatasetDict,
        split: str = "validation",
        data_loader: Optional[Lazy[TangoDataLoader]] = None,
    ) -> EvaluationResult:
        """
        Runs an evaluation on a dataset.

        * `model` is the model we want to evaluate.
        * `dataset` is the dataset we want to evaluate on.
        * `split` is the name of the split we want to evaluate on.
        * `data_loader` gives you the chance to choose a custom dataloader for the evaluation.
          By default this step evaluates on batches of 32 instances each.
        """

        concrete_data_loader: TangoDataLoader
        if data_loader is None:
            concrete_data_loader = BatchSizeDataLoader(
                dataset.splits[split], batch_size=32, shuffle=False
            )
        else:
            concrete_data_loader = data_loader.construct(instances=dataset.splits[split])

        if torch.cuda.device_count() > 0:
            cuda_device = torch.device(0)
        else:
            cuda_device = torch.device("cpu")
        check_for_gpu(cuda_device)

        generator_tqdm = Tqdm.tqdm(iter(concrete_data_loader))

        # Number of batches in instances.
        predictions: List[Dict[str, Any]] = []
        # Number of batches where the model produces a loss.
        loss_count = 0
        batch_count = 0
        # Cumulative loss
        total_loss = 0.0

        with torch.inference_mode():
            model.eval()

            for batch in concrete_data_loader:
                batch_count += 1
                batch = move_to_device(batch, cuda_device)
                output_dict = model(**batch)

                metrics = model.get_metrics()

                loss = output_dict.pop("loss", None)
                if loss is not None:
                    loss_count += 1
                    total_loss += loss.item()
                    metrics["loss"] = total_loss / loss_count

                    if any(metric_name.startswith("_") for metric_name in metrics):
                        self.logger.warning_once(
                            'Metrics with names beginning with "_" will '
                            "not be logged to the tqdm progress bar."
                        )

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

                output_dict = sanitize(output_dict)

                # This is write-only code, but it's quite fast.
                predictions.extend(
                    dict(zip(output_dict.keys(), x)) for x in zip(*output_dict.values())
                )

            final_metrics = model.get_metrics(reset=True)

        if loss_count > 0:
            # Sanity check
            if loss_count != batch_count:
                raise RuntimeError(
                    "The model you are trying to evaluate only sometimes produced a loss!"
                )
            final_metrics["loss"] = total_loss / loss_count

        return self.EvaluationResult(final_metrics, predictions)
