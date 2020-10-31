from allennlp.training.deepspeed.trainer import DeepspeedTrainer
from allennlp.training.deepspeed.optimizers import (
    FusedAdamOptimizer,
    DeepspeedCPUAdamOptimizer,
    FusedLambOptimizer
)

try:
    from allennlp.training.deepspeed.sparse_transformer_embedder import SparseTransformerEmbedder
except ImportError:
    pass