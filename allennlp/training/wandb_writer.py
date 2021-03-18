import os
from typing import Optional, Dict, Any, List, Union

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.models.model import Model
from allennlp.training.log_writer import LogWriter
from allennlp.training.optimizers import Optimizer


@LogWriter.register("wandb")
class WandBWriter(LogWriter):
    """
    Logs training runs to Weights & Biases.

    Requires the environment variable 'WANDB_API_KEY' to be set.
    """

    def __init__(
        self,
        serialization_dir: str,
        model: Model,
        optimizer: Optimizer,
        summary_interval: int = 100,
        distribution_interval: Optional[int] = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        watch_model: bool = True,
    ) -> None:
        if "WANDB_API_KEY" not in os.environ:
            raise ValueError("Missing environment variable 'WANDB_API_KEY'")

        super().__init__(
            serialization_dir,
            model,
            optimizer,
            summary_interval=summary_interval,
            distribution_interval=distribution_interval,
            batch_size_interval=batch_size_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        import wandb

        self.wandb = wandb
        config_path = os.path.join(self._serialization_dir, "config.json")
        self.wandb.init(
            project=project,
            config=Params.from_file(config_path).as_dict(),
            tags=tags,
        )
        self.wandb.save(config_path)
        if watch_model:
            self.wandb.watch(model)

    @overrides
    def log_scalars(
        self,
        scalars: Dict[str, Union[int, float]],
        log_prefix: str = "",
        epoch: Optional[int] = None,
    ) -> None:
        self._log(scalars, log_prefix=log_prefix, epoch=epoch)

    @overrides
    def log_tensors(
        self, tensors: Dict[str, torch.Tensor], log_prefix: str = "", epoch: Optional[int] = None
    ) -> None:
        self._log(
            {k: self.wandb.Histogram(v.cpu().data.numpy().flatten()) for k, v in tensors.items()},
            log_prefix=log_prefix,
            epoch=epoch,
        )

    def _log(
        self, dict_to_log: Dict[str, Any], log_prefix: str = "", epoch: Optional[int] = None
    ) -> None:
        if log_prefix:
            dict_to_log = {f"{log_prefix}/{k}": v for k, v in dict_to_log.items()}
        if epoch is not None:
            dict_to_log["epoch"] = epoch
            self.wandb.log(dict_to_log)
        else:
            self.wandb.log(dict_to_log, step=self._batch_num_total)

    @overrides
    def close(self) -> None:
        """
        Close out and log final metrics.
        """
        super().close()
        for name in ("metrics.json", "out.log"):
            path = os.path.join(self._serialization_dir, name)
            if os.path.exists(path):
                self.wandb.save(path)
