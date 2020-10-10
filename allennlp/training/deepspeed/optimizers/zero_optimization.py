from deepspeed.runtime.zero.stage2 import FP16_DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage1 import FP16_DeepSpeedZeroOptimizer_Stage1
from deepspeed.runtime.zero.utils import is_zero_supported_optimizer
# from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
# from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
# from deepspeed.runtime.config import (
#     DeepSpeedConfig,
#     ADAM_OPTIMIZER, 
#     LAMB_OPTIMIZER, 
#     ONEBIT_ADAM_OPTIMIZER, 
#     DEEPSPEED_ADAM, 
#     DEEPSPEED_OPTIMIZERS
# )

from allennlp.common import Registrable, Lazy
from allennlp.training.optimizers import Optimizer



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



class ZeroOptimizer(Registrable):
    default_implementation = "stage_2" # "disabled"

@ZeroOptimizer.register('stage_1', constructor='construct')
class ZeroStage1Optimizer(ZeroOptimizer, FP16_DeepSpeedZeroOptimizer_Stage1):
    stage = 1

    @staticmethod
    def construct(
        init_optimizer: Optimizer,
        dp_process_group=None,
        mpu=None,
        **kwargs,
    ):
        return FP16_DeepSpeedZeroOptimizer_Stage1(
            init_optimizer,
            timers=timers, 
            dp_process_group=dp_process_group, 
            mpu=mpu, 
            **kwargs
        )


@ZeroOptimizer.register('stage_2', constructor='construct')
class ZeroStage2Optimizer(ZeroOptimizer, FP16_DeepSpeedZeroOptimizer):
    stage = 2

    @staticmethod
    def construct(
        init_optimizer: Optimizer,
        timers = DummyTimer(),
        dp_process_group=None,
        mpu=None,
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        dynamic_loss_args=None,
        verbose=False,
        contiguous_gradients=True,
        reduce_bucket_size=500000000,
        allgather_bucket_size=5000000000,
        reduce_scatter=True,
        overlap_comm=False,
        cpu_offload=False,
        clip_grad=0.0,
        allreduce_always_fp32=False,
        postscale_gradients=True,
        gradient_predivide_factor=1.0,
        gradient_accumulation_steps=1
    ):
        return FP16_DeepSpeedZeroOptimizer(
            init_optimizer,
            timers=timers, 
            dp_process_group=dp_process_group, 
            mpu=mpu, 
            dynamic_loss_scale=dynamic_loss_scale,
            dynamic_loss_args=dynamic_loss_args,
            verbose=verbose,
            contiguous_gradients=contiguous_gradients,
            reduce_bucket_size=reduce_bucket_size,
            allgather_bucket_size=allgather_bucket_size,
            reduce_scatter=reduce_scatter,
            overlap_comm=overlap_comm,
            cpu_offload=cpu_offload,
            clip_grad=clip_grad,
            allreduce_always_fp32=allreduce_always_fp32,
            postscale_gradients=postscale_gradients,
            gradient_predivide_factor=gradient_predivide_factor,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

# @ZeroOptimizer.register('stage_2')
# class ZeroStage2Optimizer(FP16_DeepSpeedZeroOptimizer):
#     def __init__(self, init_optimizer=None, timers=DummyTimer(), **kwargs):
#         print('!!!!!!!!!!!!!!!')
#         print(kwargs)
#         assert init_optimizer is not None, init_optimizer
#         super().__init__(init_optimizer, timers=timers, **kwargs)


# ZeroOptimizer.register('stage_1')(FP16_DeepSpeedZeroOptimizer_Stage1)
# ZeroOptimizer.register('stage_2')(FP16_DeepSpeedZeroOptimizer)