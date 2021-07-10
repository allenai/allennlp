import dataclasses
from typing import Dict, Any, List, Optional

import torch

from allennlp.common import Lazy, Tqdm
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import sanitize
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from allennlp.tango.dataloader import TangoDataLoader, BatchSizeDataLoader
from allennlp.tango.dataset import AllenNlpDataset
from allennlp.tango.step import Step


@Step.register("evaluation")
class EvaluationStep(Step):
    DETERMINISTIC = True
    VERSION = "001"

    @dataclasses.dataclass
    class EvaluationResult:
        metrics: Dict[str, Any]
        predictions: List[
            Dict[str, Any]
        ]  # TODO: This does not make sense as a type. Should be a List with one element per instance?

    def run(  # type: ignore
        self,
        model: Model,
        dataset: AllenNlpDataset,
        split: str = "validation",
        data_loader: Optional[Lazy[TangoDataLoader]] = None,
    ):
        concrete_data_loader: TangoDataLoader
        if data_loader is None:
            concrete_data_loader = BatchSizeDataLoader(dataset.splits[split], 32, shuffle=False)
        else:
            concrete_data_loader = data_loader.construct(instances=dataset.splits[split])

        if torch.cuda.device_count() > 0:
            cuda_device = torch.device(0)
        else:
            cuda_device = torch.device("cpu")
        check_for_gpu(cuda_device)

        generator_tqdm = Tqdm.tqdm(iter(concrete_data_loader))

        # Number of batches in instances.
        batch_results = []
        # Number of batches where the model produces a loss.
        loss_count = 0
        # Cumulative loss
        total_loss = 0.0

        with torch.no_grad():
            model.eval()

            for batch in concrete_data_loader:
                batch = move_to_device(batch, cuda_device)
                output_dict = model(**batch)
                batch_results.append(sanitize(output_dict))

                metrics = model.get_metrics()

                loss = output_dict.get("loss")
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

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            # Sanity check
            if loss_count != len(batch_results):
                raise RuntimeError(
                    "The model you are trying to evaluate only sometimes produced a loss!"
                )
            final_metrics["loss"] = total_loss / loss_count

        return self.EvaluationResult(final_metrics, output_dict)
