from deepspeed.runtime.zero.utils import is_zero_supported_optimizer
from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.config import (
    DeepSpeedConfig,
    ADAM_OPTIMIZER, 
    LAMB_OPTIMIZER, 
    ONEBIT_ADAM_OPTIMIZER, 
    DEEPSPEED_ADAM, 
    DEEPSPEED_OPTIMIZERS
)

from apex.optimizers.fused_adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.lamb import FusedLamb

# from allennlp.common import Registrable
from allennlp.training.optimizers import Optimizer

# class DeepspeedOptimizer(Registrable):
#     default_implementation = "fused_adam"

# DeepspeedOptimizer.register('adam_cpu')(FP16_DeepSpeedZeroOptimizer_Stage1)
# DeepspeedOptimizer.register('fused_adam')(FusedAdam)
# DeepspeedOptimizer.register('deepspeed_adam')(DeepSpeedCPUAdam)
# DeepspeedOptimizer.register('one_bit_adam')
# DeepspeedOptimizer.register('lamb')

# Optimizer.register('adam_cpu')(FP16_DeepSpeedZeroOptimizer_Stage1)
# Optimizer.register('fused_adam')(FusedAdam)
# Optimizer.register('deepspeed_cpu_adam')(DeepSpeedCPUAdam)
# Optimizer.register('lamb')(FusedLamb)

@Optimizer.register('fused_adam', constructor='construct')
class DeepspeedFusedAdamOptimizer(Optimizer, FusedAdam):
    @staticmethod
    def construct(model_parameters, **kwargs):
        return FusedAdam(model_parameters, **kwargs)

try:
    from deepspeed.runtime.fp16.onebit_adam import OnebitAdam
    Optimizer.register('one_bit_adam')(OnebitAdam)
except ImportError:
    pass