from allennlp.training.optimizers.optimizer import (
    ParameterGroupsType,
    make_parameter_groups,
    Optimizer,
    MultiOptimizer,
    AdamOptimizer,
    SparseAdamOptimizer,
    AdamaxOptimizer,
    AdamWOptimizer,
    HuggingfaceAdamWOptimizer,
    AdagradOptimizer,
    AdadeltaOptimizer,
    SgdOptimizer,
    RmsPropOptimizer,
    AveragedSgdOptimizer,
    DenseSparseAdam,
)

try:
    from allennlp.training.optimizers.fairscale_oss import (
        FairScaleOssOptimizer,
    )
except ModuleNotFoundError as exc:
    if exc.name == "fairscale":
        pass
    else:
        raise
