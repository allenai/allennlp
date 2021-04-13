import os
from typing import Dict, Union, Optional

from overrides import overrides
from tensorboardX import SummaryWriter
import torch

from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.callbacks.log_writer import LogWriterCallback


@TrainerCallback.register("tensorboard")
class TensorBoardCallback(LogWriterCallback):
    """
    A callback that writes training statistics/metrics to TensorBoard.
    """

    def __init__(
        self,
        serialization_dir: str,
        summary_interval: int = 100,
        distribution_interval: Optional[int] = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
    ) -> None:
        super().__init__(
            serialization_dir,
            summary_interval=summary_interval,
            distribution_interval=distribution_interval,
            batch_size_interval=batch_size_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        # Create log directories prior to creating SummaryWriter objects
        # in order to avoid race conditions during distributed training.
        train_ser_dir = os.path.join(self.serialization_dir, "log", "train")
        os.makedirs(train_ser_dir, exist_ok=True)
        self._train_log = SummaryWriter(train_ser_dir)
        val_ser_dir = os.path.join(self.serialization_dir, "log", "validation")
        os.makedirs(val_ser_dir, exist_ok=True)
        self._validation_log = SummaryWriter(val_ser_dir)

    @overrides
    def log_scalars(
        self,
        scalars: Dict[str, Union[int, float]],
        log_prefix: str = "",
        epoch: Optional[int] = None,
    ) -> None:
        timestep = epoch if epoch is not None else self.trainer._batch_num_total  # type: ignore[union-attr]
        log = self._train_log if not log_prefix.startswith("validation") else self._validation_log
        for key, value in scalars.items():
            name = f"{log_prefix}/{key}" if log_prefix else key
            log.add_scalar(name, value, timestep + 1)

    @overrides
    def log_tensors(
        self, tensors: Dict[str, torch.Tensor], log_prefix: str = "", epoch: Optional[int] = None
    ) -> None:
        timestep = epoch if epoch is not None else self.trainer._batch_num_total  # type: ignore[union-attr]
        log = self._train_log if not log_prefix.startswith("validation") else self._validation_log
        for key, values in tensors.items():
            name = f"{log_prefix}/{key}" if log_prefix else key
            values_to_write = values.cpu().data.numpy().flatten()
            log.add_histogram(name, values_to_write, timestep + 1)

    @overrides
    def close(self) -> None:
        """
        Calls the `close` method of the `SummaryWriter` s which makes sure that pending
        scalars are flushed to disk and the tensorboard event files are closed properly.
        """
        super().close()
        if self._train_log is not None:
            self._train_log.close()
        if self._validation_log is not None:
            self._validation_log.close()
