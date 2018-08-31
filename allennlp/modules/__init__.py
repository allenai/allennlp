"""
Custom PyTorch
`Module <http://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ s
that are used as components in AllenNLP
:class:`~allennlp.models.model.Model` s.
"""

from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.elmo import Elmo
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.highway import Highway
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.maxout import Maxout
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.pruner import Pruner
# TODO(mattg): This line is to keep backwards compatibility.  I'm not sure how to give a
# deprecation warning on using this import, but we should remove this around version 0.8.
from allennlp.modules.pruner import Pruner as SpanPruner
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders import TokenEmbedder, Embedding
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.modules.attention import Attention
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.bimpm_matching import BiMpmMatching
