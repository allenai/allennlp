import logging
import os
from typing import Optional, Dict, Any, List, Union, Tuple, TYPE_CHECKING

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.callbacks.log_writer import LogWriterCallback


if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


logger = logging.getLogger(__name__)


@TrainerCallback.register("wandb")
class WandBCallback(LogWriterCallback):
    """
    Logs training runs to Weights & Biases.

    !!! Note
        This requires the environment variable 'WANDB_API_KEY' to be set in order
        to authenticate with Weights & Biases. If not set, you may be prompted to
        log in or upload the experiment to an anonymous account.

    In addition to the parameters that `LogWriterCallback` takes, there are several other
    parameters specific to `WandBWriter` listed below.

    # Parameters

    project : `Optional[str]`, optional (default = `None`)
        The name of the W&B project to save the training run to.
    entity : `Optional[str]`, optional (default = `None`)
        The username or team name to send the run to. If not specified, the default
        will be used.
    group : `Optional[str]`, optional (default = `None`)
        Specify a group to organize individual runs into a larger experiment.
    name : `Optional[str]`, optional (default = `None`)
        A short display name for this run, which is how you'll identify this run in the W&B UI.
        By default a random name is generated.
    notes : `Optional[str]`, optional (default = `None`)
        A description of the run.
    tags : `Optional[List[str]]`, optional (default = `None`)
        Tags to assign to the training run in W&B.
    watch_model : `bool`, optional (default = `True`)
        Whether or not W&B should watch the `Model`.
    files_to_save : `Tuple[str, ...]`, optional (default = `("config.json", "out.log")`)
        Extra files in the serialization directory to save to the W&B training run.
    wandb_kwargs : `Optional[Dict[str, Any]]`, optional (default = `None`)
        Additional key word arguments to pass to [`wandb.init()`](https://docs.wandb.ai/ref/python/init).
    """

    def __init__(
        self,
        serialization_dir: str,
        summary_interval: int = 100,
        distribution_interval: Optional[int] = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        watch_model: bool = True,
        files_to_save: Tuple[str, ...] = ("config.json", "out.log"),
        wandb_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if "WANDB_API_KEY" not in os.environ:
            logger.warning(
                "Missing environment variable 'WANDB_API_KEY' required to authenticate to Weights & Biases."
            )

        super().__init__(
            serialization_dir,
            summary_interval=summary_interval,
            distribution_interval=distribution_interval,
            batch_size_interval=batch_size_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        self._watch_model = watch_model
        self._files_to_save = files_to_save
        self._wandb_kwargs: Dict[str, Any] = dict(
            dir=os.path.abspath(serialization_dir),
            project=project,
            entity=entity,
            group=group,
            name=name,
            notes=notes,
            config=Params.from_file(os.path.join(serialization_dir, "config.json")).as_dict(),
            tags=tags,
            anonymous="allow",
            **(wandb_kwargs or {}),
        )

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
            {k: self.wandb.Histogram(v.cpu().data.numpy().flatten()) for k, v in tensors.items()},  # type: ignore
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
        self.wandb.log(dict_to_log, step=self.trainer._total_batches_completed)  # type: ignore

    @overrides
    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary=is_primary, **kwargs)

        if not is_primary:
            return None

        import wandb

        self.wandb = wandb
        self.wandb.init(**self._wandb_kwargs)

        for fpath in self._files_to_save:
            self.wandb.save(  # type: ignore
                os.path.join(self.serialization_dir, fpath), base_path=self.serialization_dir
            )

        if self._watch_model:
            self.wandb.watch(self.trainer.model)  # type: ignore

    @overrides
    def close(self) -> None:
        super().close()
        self.wandb.finish()  # type: ignore
