"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

from typing import Optional, Union, List

import torch
import re

from allennlp.common import Lazy
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import log_frozen_and_tunable_parameter_names
from allennlp.models import Model
from allennlp.tango.dataloader import TangoDataLoader, MaxBatchesDataLoader, DataLoaderAdapter
from allennlp.tango.dataset import DatasetDict
from allennlp.tango.format import TorchFormat, Format
from allennlp.tango.step import Step
from allennlp.training import Checkpointer, TrainerCallback, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer


@Step.register("training")
class TrainingStep(Step):
    """This step trains a model given the model, the dataset, and various hyperparameters."""

    DETERMINISTIC = True
    VERSION = "003"
    FORMAT: Format = TorchFormat()

    # Development notes:
    #
    # This is not taking a cuda_device. We autodetect those. If you don't want to run with the GPU, set
    # the CUDA_DEVICES environment variable to be empty.
    #
    # This is adaptering so we can use the original trainer. But the original trainer API is insane. You
    # instantiate the object, and then you can call exactly one method on it (.train()), and you can
    # call it exactly once. If you do anything else crazy things happen. We should replace the trainer API
    # entirely and transplant the logic from the .train() method directly into the step's .run() method.
    # If we do want to have a separate Trainer object, it should take data loaders and models in the .train()
    # method, not in __init__(), and allow multiple calls to that method (even multiple concurrent ones). That
    # would be a sane API.

    def run(  # type: ignore
        self,
        model: Lazy[Model],
        dataset: DatasetDict,
        data_loader: Lazy[TangoDataLoader],
        optimizer: Lazy[Optimizer],
        validation_data_loader: Optional[Lazy[TangoDataLoader]] = None,
        training_split: str = "train",
        validation_split: Optional[str] = None,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        checkpointer: Optional[Lazy[Checkpointer]] = None,
        grad_norm: Union[float, bool] = False,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[Lazy[LearningRateScheduler]] = None,
        momentum_scheduler: Optional[Lazy[MomentumScheduler]] = None,
        moving_average: Optional[Lazy[MovingAverage]] = None,
        callbacks: List[Lazy[TrainerCallback]] = None,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        enable_default_callbacks: bool = True,
        run_confidence_checks: bool = True,
        no_grad: Optional[List[str]] = None,
        limit_batches_per_epoch: Optional[int] = None,
    ) -> Model:
        serialization_dir = self.work_dir()

        if validation_data_loader is None:
            validation_data_loader = data_loader
        if validation_split is None:
            validation_loader = None
        else:
            concrete_validation_data_loader = validation_data_loader.construct(
                instances=dataset.splits[validation_split]
            )
            del validation_data_loader
            if limit_batches_per_epoch is not None:
                concrete_validation_data_loader = MaxBatchesDataLoader(
                    concrete_validation_data_loader, limit_batches_per_epoch
                )
            validation_loader = DataLoaderAdapter(tango_data_loader=concrete_validation_data_loader)

        concrete_data_loader = data_loader.construct(instances=dataset.splits[training_split])
        del data_loader
        if limit_batches_per_epoch is not None:
            concrete_data_loader = MaxBatchesDataLoader(
                concrete_data_loader, limit_batches_per_epoch
            )
        loader = DataLoaderAdapter(tango_data_loader=concrete_data_loader)

        if torch.cuda.device_count() > 0:
            cuda_device = torch.device(0)
        else:
            cuda_device = torch.device("cpu")
        check_for_gpu(cuda_device)
        loader.set_target_device(cuda_device)
        if validation_loader is not None:
            validation_loader.set_target_device(cuda_device)

        concrete_model = model.construct(vocab=dataset.vocab).to(cuda_device)
        del model
        if no_grad:
            for name, parameter in concrete_model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)
        parameters = [[n, p] for n, p in concrete_model.named_parameters() if p.requires_grad]
        concrete_optimizer = optimizer.construct(model_parameters=parameters)
        del optimizer
        log_frozen_and_tunable_parameter_names(concrete_model)

        concrete_moving_average = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        del moving_average

        concrete_learning_rate_scheduler = (
            None
            if learning_rate_scheduler is None
            else learning_rate_scheduler.construct(
                optimizer=concrete_optimizer,
                num_epochs=num_epochs,
                num_steps_per_epoch=concrete_data_loader.num_batches_per_epoch(),
            )
        )
        del learning_rate_scheduler

        concrete_momentum_scheduler = (
            None
            if momentum_scheduler is None
            else momentum_scheduler.construct(optimizer=concrete_optimizer)
        )
        del momentum_scheduler

        if checkpointer is not None:
            concrete_checkpointer = checkpointer.construct(serialization_dir=serialization_dir)
        else:
            concrete_checkpointer = Checkpointer(serialization_dir)
        del checkpointer

        concrete_callbacks: List[TrainerCallback] = [
            cb.construct(serialization_dir=serialization_dir) for cb in callbacks or []
        ]
        del callbacks

        trainer = GradientDescentTrainer(
            concrete_model,
            optimizer=concrete_optimizer,
            data_loader=loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            checkpointer=concrete_checkpointer,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=concrete_learning_rate_scheduler,
            momentum_scheduler=concrete_momentum_scheduler,
            moving_average=concrete_moving_average,
            callbacks=concrete_callbacks,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            enable_default_callbacks=enable_default_callbacks,
            run_confidence_checks=run_confidence_checks,
        )
        trainer.train()

        return trainer.model
