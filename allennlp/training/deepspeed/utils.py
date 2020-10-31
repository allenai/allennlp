import logging
from deepspeed.utils import logger as ds_logger
ds_logger.setLevel(logging.WARNING)
ds_logger.propagate = False

import torch
from allennlp.models.model import Model
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.training.deepspeed.config import DeepspeedConfig, DeepspeedArgs

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine

def launch_deepspeed(
    model: Model,
    optimizer: torch.optim.Optimizer,
    config: DeepspeedConfig,
    args: Lazy[DeepspeedArgs],
    batch_size: int,
    gradient_accumulation_steps: int,
):
    if not(optimizer is None or config.optimizer is None):
        raise ConfigurationError(f"Cannot provide both optimizer and deepspeed_optimizer. {optimizer, config.to_dict()}")

    config = dict(**config.to_dict(), train_batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps)
    ds = DeepSpeedEngine(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        dist_init_required=False,
        config_params=config
    )
    if hasattr(ds, 'timers'):
        def mute_log(*args, **kwargs):
            pass
        ds.timers.log = mute_log
    return ds, ds.optimizer