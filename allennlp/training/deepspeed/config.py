from typing import Dict, Any
from enum import IntEnum
from allennlp.common import FromParams
from dataclasses import dataclass, asdict


@dataclass
class DeepspeedFP16Config(FromParams):
    enabled: bool = True
    loss_scale: float = 0.
    initial_scale_power: int = 32
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: float = 1.

@dataclass
class DeepspeedAMPConfig(FromParams):
    enabled: bool = False
    opt_level: str = "O1"

@dataclass
class DeepspeedOptimizerConfig(FromParams):
    type: str
    params: Dict[str, Any]

class DeepspeedZeROStage(IntEnum):
    DISABLED = 0
    OPTIMIZER = 1
    GRADIENT = 2

@dataclass
class DeepspeedZeROConfig(FromParams):
    stage: DeepspeedZeROStage = DeepspeedZeROStage.GRADIENT
    allgather_partitions: bool = True
    allgather_bucket_size: int = 500000000
    overlap_comm: bool = False
    reduce_scatter: bool = True
    reduce_bucket_size: int = 500000000
    contiguous_gradients: bool = False
    cpu_offload: bool = False


@dataclass
class DeepspeedConfig(FromParams):
    zero_optimization: DeepspeedZeROConfig
    fp16: DeepspeedFP16Config
    amp: DeepspeedAMPConfig = DeepspeedAMPConfig()
    optimizer: DeepspeedOptimizerConfig = None
    
    zero_allow_untested_optimizer: bool = True
    wall_clock_breakdown: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class DeepspeedArgs(FromParams):
    local_rank: int
    deepspeed: bool = True
    deepspeed_mpi: bool = False
    deepspeed_config: str = None