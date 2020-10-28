'''
Copyright 2019 The Microsoft DeepSpeed Team
'''

import logging
from deepspeed.utils import logger as ds_logger
ds_logger.setLevel(logging.WARNING)
ds_logger.propagate = False

import os
import torch
import warnings
import torch.distributed as dist

import apex
from apex.optimizers import (
    FusedAdam,
    FusedLAMB
)
from torch import nn
from torch.distributed.distributed_c10d import _get_global_rank

from deepspeed.runtime.zero.stage2 import FP16_DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage1 import FP16_DeepSpeedZeroOptimizer_Stage1
from deepspeed.runtime.zero.utils import is_zero_supported_optimizer
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.fp16.onebit_adam import OnebitAdam
from deepspeed.runtime.config import DeepSpeedConfig, \
    ADAM_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, DEEPSPEED_ADAM, DEEPSPEED_OPTIMIZERS
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.constants import \
    ROUTE_TRAIN, ROUTE_PREDICT, ROUTE_EVAL, \
    TORCH_DISTRIBUTED_DEFAULT_PORT
from deepspeed.runtime.zero.constants import \
    ZERO_OPTIMIZATION_OPTIMIZER_STATES, ZERO_OPTIMIZATION_GRADIENTS
from deepspeed.runtime.csr_tensor import CSRTensor
import deepspeed.runtime.lr_schedules as lr_schedules
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.engine import (
    _initialize_parameter_parallel_groups, 
    split_half_float_double_csr,
    flatten,
    unflatten,
    MEMORY_OPT_ALLREDUCE_SIZE
)

from allennlp.common import Lazy, FromParams
from allennlp.training.deepspeed.optimizers.zero_optimization import ZeroOptimizer
from allennlp.training.deepspeed.optimizers.basic import *


class DummyTimer:
    class Timer:
        def __init__(self, name):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def reset(self):
            pass

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, *args, **kwargs):
        pass




class AllennlpDeepSpeedEngineAdapter(FromParams, nn.Module):
    r"""DeepSpeed engine for training.
    """
    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 zero_optimizer: Lazy[ZeroOptimizer] = None,
                 model_parameters=None,
                 lr_scheduler=None,
                 mpu=None,
                 dist_init_required=None,
                 config_params=None
        ):
        super().__init__()
        self.zero_optimizer = zero_optimizer

        self.client_optimizer = optimizer
        self.client_model_parameters = model_parameters
        self.client_lr_scheduler = lr_scheduler
        self.mpu = mpu
        self.data_parallel_group = None
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gradient_average = True
        self.warn_unscaled_loss = True
        self.config_params = config_params
        self.enable_backward_allreduce = True

        if dist_init_required is None:
            dist_init_required = not dist.is_initialized()

        self.dist_backend = "nccl"
        if dist_init_required:
            if not dist.is_initialized():
                logger.info("Initializing torch distributed with backend: {}".format(
                    self.dist_backend))
                dist.init_process_group(backend=self.dist_backend)
            else:
                logger.warning(
                    "Was given dist_init_required=True but detected that torch"
                    "distributed was already initialized, cannot initialize twice.")

        self._configure_with_arguments(args, mpu)

        self._init_distributed(dist_init_required)

        # Configure distributed model
        self._configure_distributed_model(model)

        # Configure optimizer and scheduler
        self.optimizer = self._configure_optimizer(optimizer, model_parameters)
        self._configure_lr_scheduler(lr_scheduler)

        # Bookkeeping for csr support
        self.csr_tensor_module_names = set()
        if self.sparse_gradients_enabled:
            for name, module in self.module.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    self.csr_tensor_module_names.add(name + ".weight")

    @property
    def dynamic_loss_scale(self):
        return self.loss_scale == 0

    @property
    def postscale_gradients(self):
        return not self._config.prescale_gradients

    def _configure_lr_scheduler(self, client_lr_scheduler):
        # First check for scheduler in json configuration
        lr_scheduler = self._scheduler_from_config(self.optimizer)
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler
        else:
            self.lr_scheduler = client_lr_scheduler

    def _scheduler_from_config(self, optimizer):
        scheduler_name = self.scheduler_name
        if scheduler_name is not None:
            if hasattr(lr_schedules, scheduler_name):
                scheduler = getattr(lr_schedules, scheduler_name)
            else:
                assert hasattr(torch.optim.lr_scheduler, scheduler_name), \
                    f"DeepSpeed does not recognize LR scheduler {scheduler_name}"

                scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)

            instantiated_scheduler = scheduler(optimizer, **self.scheduler_params)
            return instantiated_scheduler
        else:
            return None

    def _init_distributed(self, dist_init_required):
        if self.local_rank >= 0:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = torch.device("cuda")

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        self.local_rank = args.local_rank if hasattr(args, 'local_rank') else 0
        self._config = DeepSpeedConfig(args.deepspeed_config,
                                       mpu,
                                       param_dict=self.config_params)
        for k, v in vars(self._config).items():
            setattr(self, k, v)


    def _is_supported_optimizer(self, optimizer_name):
        return optimizer_name in DEEPSPEED_OPTIMIZERS or \
            getattr(torch.optim, optimizer_name, None) is not None

    def _broadcast_model(self):
        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p,
                               self.broadcast_src_rank,
                               group=self.data_parallel_group)

    def _configure_distributed_model(self, model):
        self.module = model
        if self.fp16_enabled:
            self.module.half()
        self.module.to(self.device)

        if self.mpu is None:
            self.data_parallel_group = _initialize_parameter_parallel_groups()
            self.dp_world_size = dist.get_world_size()
            self.mp_world_size = 1
            self.broadcast_src_rank = 0
        else:
            self.data_parallel_group = self.mpu.get_data_parallel_group()
            self.dp_world_size = self.mpu.get_data_parallel_world_size()
            self.mp_world_size = self.mpu.get_model_parallel_world_size()
            self.broadcast_src_rank = _get_global_rank(
                self.mpu.get_data_parallel_group(),
                0
            )

        self._broadcast_model()

    def _configure_optimizer(self, client_optimizer, model_parameters):
        basic_optimizer = client_optimizer

        if self.zero_enabled: #zero_optimization: # self.zero_optimizer or
            if not is_zero_supported_optimizer(basic_optimizer):
                assert self.zero_allow_untested_optimizer, \
                    'You are using an untested ZeRO Optimizer. Please add <"zero_allow_untested_optimizer": true> in the configuration file to use it.'

                if self.global_rank == 0:
                    logger.warning("**** You are using ZeRO with an untested optimizer, proceed with caution *****")
            
            return self._configure_zero_optimizer(basic_optimizer)

        if self.fp16_enabled:
            return self._configure_fp16_optimizer(basic_optimizer)
        
        return basic_optimizer


    def _configure_fp16_optimizer(self, optimizer):
        defaults = dict(
            init_optimizer=optimizer,
            mpu=self.mpu,
            clip_grad=self.gradient_clipping,
            fused_adam_legacy=self.optimizer_legacy_fusion,
            timers=None,
            verbose=False
        )

        if not self.dynamic_loss_scale:
            return FP16_Optimizer(**defaults, static_loss_scale=self.loss_scale)

        defaults.update(dict(
            dynamic_loss_scale=True,
            dynamic_loss_args=self.dynamic_loss_scale_args,
        ))

        if isinstance(optimizer, (FusedAdam, OnebitAdam)):
            extras = dict(initial_dynamic_scale=self.initial_dynamic_scale)
        else:
            extras = dict(fused_lamb_legacy=isinstance(optimizer, FusedLAMB))
        optimizer = FP16_Optimizer(**defaults, **extras)
        return optimizer

    def _configure_zero_optimizer(self, optimizer):
        optimizer = self.zero_optimizer.construct(
            init_optimizer=optimizer,
            dp_process_group=self.data_parallel_group,
            mpu=self.mpu
        )
        assert not (isinstance(optimizer, FP16_DeepSpeedZeroOptimizer_Stage1) and not self.zero_reduce_scatter), 'Stage 1 only supports reduce scatter mode'
        return optimizer

    def train(self):
        r"""
        """

        self.warn_unscaled_loss = True
        self.module.train()

    def eval(self):
        r"""
        """

        self.warn_unscaled_loss = True
        self.module.train(False)

    def _scale_loss(self, prescaled_loss):
        if isinstance(prescaled_loss, torch.Tensor):
            scaled_loss = prescaled_loss / self.gradient_accumulation_steps
        elif isinstance(prescaled_loss, tuple) or isinstance(prescaled_loss, list):
            scaled_loss = []
            for l in prescaled_loss:
                if isinstance(l, torch.Tensor):
                    scaled_loss.append(l / self.gradient_accumulation_steps)
                else:
                    scaled_loss.append(l)
        else:
            scaled_loss = prescaled_loss
            if self.warn_unscaled_loss:
                logger.warning(
                    f'DeepSpeed unable to scale loss because of type: {type(prescaled_loss)}'
                )
                self.warn_unscaled_loss = False

        return scaled_loss

    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        loss = self.module(*inputs, **kwargs)
        return loss

    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        #Zero stage 2 communicates during non gradient accumulation boundaries as well
        if self.zero_optimization_stage >= ZERO_OPTIMIZATION_GRADIENTS:
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        #Communicate only at gradient accumulation boundaries
        elif self.is_gradient_accumulation_boundary:
            if self.zero_optimization_stage == ZERO_OPTIMIZATION_OPTIMIZER_STATES:
                assert self.zero_reduce_scatter
                self.optimizer.reduce_scatter_gradients(
                    postscale_gradients=self.postscale_gradients,
                    gradient_predivide_factor=self.gradient_predivide_factor,
                    gradient_average=self.gradient_average)
            else:
                self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    def backward(self, loss, allreduce_gradients=True, release_loss=False):
        r"""Execute backward pass on the loss

        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: If this is False, then gradient averaging will be skipped. Default is True.
        """

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps > 1:
            loss = self._scale_loss(loss.float())

        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        if self.zero_enabled: #zero_optimization:
            self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary
            self.optimizer.backward(loss)
        elif self.fp16_enabled:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        if allreduce_gradients and self.enable_backward_allreduce:
            self.allreduce_gradients()

        return loss

    @property
    def is_gradient_accumulation_boundary(self):
        """Query whether the current micro-batch is at the boundary of
        gradient accumulation, and thus will trigger gradient reductions and
        an optimizer step.

        Returns:
            bool: if the current step is a gradient accumulation boundary.
        """
        return (self.micro_steps + 1) % \
            self.gradient_accumulation_steps == 0

    def zero_grad(self):
        """
        Zero parameter grads.
        """
        for param_name, param in self.module.named_parameters():
            param.grad = None

    def clip_fp32_gradients(self):
        torch.nn.utils.clip_grad_norm_(parameters=self.module.parameters(),
                                       max_norm=self.gradient_clipping)

    def _take_model_step(self):
        if self.gradient_clipping > 0.0 and not self.fp16_enabled:
            self.clip_fp32_gradients()
        self.optimizer.step()

        #zero grad in basic optimizer could be unreliable and may not exhibit
        #the behaviour that we want
        if not self.zero_enabled and not self.fp16_enabled:
            self.zero_grad()
        else:
            self.optimizer.zero_grad()

        # Check overlow here since in DS fp16 optimizer, the overflow is updated in above step() function.
        overflow = False
        if hasattr(self.optimizer, 'overflow'):
            overflow = self.optimizer.overflow

        if overflow:
            self.skipped_steps += 1
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def step(self):
        r"""Execute the weight update step after forward and backward propagation
        on effective_train_batch.
        """

        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use step"

        # Update the model when we reach gradient accumulation boundaries
        if self.is_gradient_accumulation_boundary:
            self._take_model_step()

        self.micro_steps += 1

    def _get_optimizer_param(self, param_name):
        result = []
        if not self.optimizer:
            return result
        for group in self.optimizer.param_groups:
            if param_name in group:
                result.append(group[param_name])
            else:
                result.append(0.0)
        return result


    def allreduce_bucket(self, bucket):
        tensor = flatten(bucket)

        tensor_to_allreduce = tensor

        if self.allreduce_always_fp32:
            tensor_to_allreduce = tensor.float()

        if self.postscale_gradients:
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)

            dist.all_reduce(tensor_to_allreduce, group=self.data_parallel_group)

            if self.gradient_average:
                if self.gradient_predivide_factor != self.dp_world_size:
                    tensor_to_allreduce.mul_(self.gradient_predivide_factor / self.dp_world_size)
        else:
            tensor_to_allreduce.div_(self.dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=self.data_parallel_group)

        if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy(self, small_bucket):
        allreduced = self.allreduce_bucket(small_bucket)
        for buf, synced in zip(small_bucket, unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def allreduce_no_retain(self, bucket, numel_per_bucket=500000000):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket)
                small_bucket = []
                numel = 0
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket)

    def buffered_allreduce_fallback(self, grads=None, elements_per_buffer=500000000):
        grads = []
        for param_name, param in self.module.named_parameters():
            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(param.size(),
                                         dtype=param.dtype,
                                         device=param.device)
                grads.append(param.grad.data)
            else:
                grad_data = param.grad.data
                if self.sparse_gradients_enabled and param_name in self.csr_tensor_module_names:
                    grads.append(CSRTensor(grad_data))
                else:
                    grads.append(grad_data)

        split_buckets = split_half_float_double_csr(grads)

        for i, bucket_tuple in enumerate(split_buckets):
            bucket_type, bucket = bucket_tuple
            if bucket_type == CSRTensor.type():
                self.csr_allreduce_no_retain(bucket)
            else:
                self.allreduce_no_retain(bucket, numel_per_bucket=elements_per_buffer)

    def csr_allreduce_no_retain(self, bucket):
        allreduced_csrs = self.csr_allreduce_bucket(bucket)
        # Densify csr tensor and copy back to original location
        for csr in allreduced_csrs:
            dense_tensor = csr.to_dense()
            csr.orig_dense_tensor.copy_(dense_tensor)

    def csr_allreduce_bucket(self, bucket):
        csr_list = []
        for csr in bucket:
            csr_list.append(self.csr_allreduce(csr))
        return csr_list

    def csr_allreduce(self, csr):
        # Pre-divide for fp16 stability
        csr.values.div_(self.dp_world_size)

        indices_device_list = self.csr_all_gather(csr.indices)
        values_device_list = self.csr_all_gather(csr.values)

        csr.indices = torch.cat(indices_device_list)
        csr.values = torch.cat(values_device_list)
        return csr

    def csr_all_gather(self, value):
        my_size = torch.LongTensor([value.size()[0]]).to(self.device)
        all_sizes = self.all_gather_scalar(my_size)
        max_size = torch.cat(all_sizes).max()
        fill_size = (max_size - my_size)

        assert value.dim() in [1, 2]
        if value.dim() == 1:
            if fill_size > 0:
                value = torch.cat([value, value.new_zeros(fill_size)])
            tensor_list = [value.new_zeros(max_size) for _ in range(self.dp_world_size)]
        else:
            if fill_size > 0:
                value = torch.cat([value, value.new_zeros(fill_size, value.size()[1])])
            tensor_list = [
                value.new_zeros(max_size,
                                value.size()[1]) for _ in range(self.dp_world_size)
            ]

        dist.all_gather(tensor_list, value, group=self.data_parallel_group)
        tensors = []
        for dev_idx, t in enumerate(tensor_list):
            size = all_sizes[dev_idx][0]
            tensors.append(
                t.index_select(0,
                               torch.LongTensor(range(size)).to(self.device)))

        return tensors

    def all_gather_scalar(self, value):
        tensor_list = [value.new_zeros(value.size()) for _ in range(self.dp_world_size)]
        dist.all_gather(tensor_list, value, group=self.data_parallel_group)
        return tensor_list