from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.fp16.onebit_adam import OnebitAdam

from allennlp.common import Registrable, Lazy
from allennlp.training.optimizers import Optimizer


class DeepspeedFP16Optimizer(Registrable):
    default_implementation = 'fused'

@DeepspeedFP16Optimizer.register('fused', constructor='construct')
class DeepspeedFusedFP16Optimizer(DeepspeedFP16Optimizer):
    @staticmethod
    def construct(
        init_optimizer: Optimizer,
        mpu=None,
        clip_grad=0.0,
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        initial_dynamic_scale=2**32,
        dynamic_loss_args=None,
        fused_adam_legacy=False,
        timers=None,
        verbose=False
    ):
    if isinstance(optimizer, (apex.optimizers.FusedAdam, OnebitAdam)):
        pass

def _configure_fp16_optimizer(self, optimizer):
        initial_dynamic_scale = self.initial_dynamic_scale()
        dynamic_loss_args = self.dynamic_loss_scale_args()

        if isinstance(optimizer, apex.optimizers.FusedAdam) or self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
            defaults['fused_adam_legacy'] = self.optimizer_legacy_fusion()
            if self.dynamic_loss_scale():
                defaults.update(dict(
                    dynamic_loss_scale=True,
                    initial_dynamic_scale=initial_dynamic_scale,
                    dynamic_loss_args=dynamic_loss_args,
                ))
            else:
                defaults.update(dict(static_loss_scale=self.loss_scale()))
            optimizer = FP16_Optimizer(**defaults)
        else:
            optimizer = FP16_UnfusedOptimizer(
                **defaults,
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=dynamic_loss_args,
                fused_lamb_legacy=isinstance(optimizer, apex.optimizers.FusedLamb)
            )
        # raise ValueError(optimizer)
        return optimizer