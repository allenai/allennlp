from typing import Optional, Union, List

import torch
import re

from allennlp.common import Lazy
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import log_frozen_and_tunable_parameter_names
from allennlp.models import Model
from allennlp.tango.dataloader import TangoDataLoader, MaxBatchesDataLoader, DataLoaderAdapter
from allennlp.tango.dataset import AllenNlpDataset
from allennlp.tango.format import TorchFormat
from allennlp.tango.step import Step
from allennlp.training import Checkpointer, TrainerCallback, GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer


@Step.register("training")
class TrainingStep(Step):
    DETERMINISTIC = True
    VERSION = "003"
    FORMAT = TorchFormat()

    # TODO: distributed training
    # TODO: recovery of failed jobs (this should be done but needs verification)

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

    def run(
        self,
        model: Lazy[Model],
        dataset: AllenNlpDataset,
        data_loader: Lazy[TangoDataLoader],
        optimizer: Lazy[Optimizer],
        validation_data_loader: Optional[Lazy[TangoDataLoader]] = None,
        training_split: str = "train",
        validation_split: Optional[str] = None,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        checkpointer: Optional[Lazy[Checkpointer]] = None,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[Lazy[LearningRateScheduler]] = None,
        momentum_scheduler: Optional[Lazy[MomentumScheduler]] = None,
        moving_average: Optional[Lazy[MovingAverage]] = None,
        callbacks: List[Lazy[TrainerCallback]] = None,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        enable_default_callbacks: bool = True,
        run_sanity_checks: bool = True,
        no_grad: Optional[List[str]] = None,
        limit_batches_per_epoch: Optional[int] = None,
    ) -> Model:
        serialization_dir = self.temp_dir()

        if validation_data_loader is None:
            validation_data_loader = data_loader
        if validation_split is None:
            validation_loader = None
        else:
            validation_data_loader = validation_data_loader.construct(
                instances=dataset.splits[validation_split]
            )
            if limit_batches_per_epoch is not None:
                validation_data_loader = MaxBatchesDataLoader(
                    validation_data_loader, limit_batches_per_epoch
                )
            validation_loader = DataLoaderAdapter(validation_data_loader)

        data_loader = data_loader.construct(instances=dataset.splits[training_split])
        if limit_batches_per_epoch is not None:
            data_loader = MaxBatchesDataLoader(data_loader, limit_batches_per_epoch)
        loader = DataLoaderAdapter(data_loader)

        if torch.cuda.device_count() > 0:
            cuda_device = torch.device(0)
        else:
            cuda_device = torch.device("cpu")
        check_for_gpu(cuda_device)
        loader.set_target_device(cuda_device)
        if validation_loader is not None:
            validation_loader.set_target_device(cuda_device)

        model = model.construct(vocab=dataset.vocab).to(cuda_device)
        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = optimizer.construct(model_parameters=parameters)
        log_frozen_and_tunable_parameter_names(model)
        moving_average = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        learning_rate_scheduler = (
            None
            if learning_rate_scheduler is None
            else learning_rate_scheduler.construct(
                optimizer=optimizer,
                num_epochs=num_epochs,
                num_steps_per_epoch=data_loader.num_batches_per_epoch(),
            )
        )
        momentum_scheduler = (
            None
            if momentum_scheduler is None
            else momentum_scheduler.construct(optimizer=optimizer)
        )
        if checkpointer is not None:
            checkpointer = checkpointer.construct(serialization_dir=serialization_dir)
        else:
            checkpointer = Checkpointer(serialization_dir)
        callbacks: List[TrainerCallback] = [
            cb.construct(serialization_dir=serialization_dir) for cb in callbacks or []
        ]

        trainer = GradientDescentTrainer(
            model,
            optimizer=optimizer,
            data_loader=loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            checkpointer=checkpointer,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler,
            momentum_scheduler=momentum_scheduler,
            moving_average=moving_average,
            callbacks=callbacks,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            enable_default_callbacks=enable_default_callbacks,
            run_sanity_checks=run_sanity_checks,
        )
        trainer.train()

        return trainer.model
