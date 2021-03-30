"""
Custom PyTorch
`Module <https://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ s
that are used as components in AllenNLP `Model` s.
"""

from allennlp.modules.attention import Attention
from allennlp.modules.backbones import Backbone
from allennlp.modules.bimpm_matching import BiMpmMatching
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.elmo import Elmo
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.gated_sum import GatedSum
from allennlp.modules.highway import Highway
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.modules.maxout import Maxout
from allennlp.modules.residual_with_layer_dropout import ResidualWithLayerDropout
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders import TokenEmbedder, Embedding
from allennlp.modules.softmax_loss import SoftmaxLoss
