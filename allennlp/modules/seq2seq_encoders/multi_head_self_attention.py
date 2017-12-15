import torch

from torch.autograd import Variable
from torch.nn import Dropout, Linear
from torch.nn import Parameter
from torch.nn import init

from allennlp.nn.util import last_dim_softmax
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("multi_head_self_attention")
class MultiHeadSelfAttention(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

    The attention mechanism is a weighted sum of a projection V of the inputs, with respect
    to the scaled, normalised dot product of Q and K, which are also both linear projections
    of the input. This procedure is repeated for each attention head, using different parameters.

    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    attention_dim ``int``, required.
        The dimension of the query and key projections which comprise the
        dot product attention function.
    values_dim : ``int``, required.
        The dimension which the input is projected to for representing the values,
        which are combined using the attention.
    output_projection_dim : ``int``, optional (default = None)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    attention_dropout_prob : ``float``, optional (default = 0.1).
        The dropout probability applied to the normalised attention
        distributions.
    """
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        self._query_projections = Parameter(torch.FloatTensor(num_heads, input_dim, attention_dim))
        self._key_projections = Parameter(torch.FloatTensor(num_heads, input_dim, attention_dim))
        self._value_projections = Parameter(torch.FloatTensor(num_heads, input_dim, values_dim))

        self._scale = input_dim ** 0.5
        self._output_projection = Linear(num_heads * values_dim,
                                         self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Because we are doing so many torch.bmm calls, which is fast but unstable,
        # it is critically important to intitialise the parameters correctly such
        # that these matrix multiplications are well conditioned initially.
        # Without this initialisation, this (non-deterministically) produces
        # NaNs and overflows.
        init.xavier_normal(self._query_projections)
        init.xavier_normal(self._key_projections)
        init.xavier_normal(self._value_projections)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        num_heads = self._num_heads

        batch_size, timesteps, hidden_dim = inputs.size()
        if mask is None:
            mask = Variable(inputs.data.new(batch_size, timesteps).fill_(1.0))

        # Treat the queries, keys and values each as a ``num_heads`` size batch.
        # shape (num_heads, batch_size * timesteps, hidden_dim)
        inputs_per_head = inputs.repeat(num_heads, 1, 1).view(num_heads,
                                                              batch_size * timesteps,
                                                              hidden_dim)
        # Do the projections for all the heads at once.
        # Then reshape the result as though it had a
        # (num_heads * batch_size) sized batch.
        queries_per_head = torch.bmm(inputs_per_head, self._query_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        queries_per_head = queries_per_head.view(num_heads * batch_size,
                                                 timesteps,
                                                 self._attention_dim)

        keys_per_head = torch.bmm(inputs_per_head, self._key_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        keys_per_head = keys_per_head.view(num_heads * batch_size,
                                           timesteps,
                                           self._attention_dim)

        values_per_head = torch.bmm(inputs_per_head, self._value_projections)
        # shape (num_heads * batch_size, timesteps, attention_dim)
        values_per_head = values_per_head.view(num_heads * batch_size, timesteps, self._values_dim)

        # shape (num_heads * batch_size, timesteps, timesteps)
        scaled_similarities = torch.bmm(queries_per_head, keys_per_head.transpose(1, 2)) / self._scale

        # shape (num_heads * batch_size, timesteps, timesteps)
        # Normalise the distributions, using the same mask for all heads.
        attention = last_dim_softmax(scaled_similarities, mask.repeat(num_heads, 1))
        attention = self._attention_dropout(attention)
        # This is doing the following batch-wise matrix multiplication:
        # (num_heads * batch_size, timesteps, timesteps) *
        # (num_heads * batch_size, timesteps, values_dim)
        # which is equivalent to a weighted sum of the values with respect to
        # the attention distributions for each element in the num_heads * batch_size
        # dimension.
        # shape (num_heads * batch_size, timesteps, values_dim)
        outputs = torch.bmm(attention, values_per_head)

        # Reshape back to original shape (batch_size, timesteps, num_heads * values_dim)
        # Note that we _cannot_ use a reshape here, because this tensor was created
        # with num_heads being the first dimension, so reshaping naively would not
        # throw an error, but give an incorrect result.
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)

        # Project back to original input size.
        # shape (batch_size, timesteps, input_size)
        outputs = self._output_projection(outputs)
        return outputs
