from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import Activation


@MatrixAttention.register("bilinear")
class BilinearMatrixAttention(MatrixAttention):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights `W` and a bias `b`, and the similarity between the two matrices `X`
    and `Y` is computed as `X W Y^T + b`.

    Registered as a `MatrixAttention` with name "bilinear".

    # Parameters

    matrix_1_dim : `int`, required
        The dimension of the matrix `X`, described above.  This is `X.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : `int`, required
        The dimension of the matrix `Y`, described above.  This is `Y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `X W Y^T + b` calculation.  Default is
        linear, i.e. no activation.
    use_input_biases : `bool`, optional (default = `False`)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    label_dim : `int`, optional (default = `1`)
        The number of output classes. Typically in an attention setting this will be one,
        but this parameter allows this class to function as an equivalent to `torch.nn.Bilinear`
        for matrices, rather than vectors.
    """

    def __init__(
        self,
        matrix_1_dim: int,
        matrix_2_dim: int,
        activation: Activation = None,
        use_input_biases: bool = False,
        label_dim: int = 1,
    ) -> None:
        super().__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if label_dim == 1:
            self._weight_matrix = Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = Parameter(torch.Tensor(label_dim, matrix_1_dim, matrix_2_dim))

        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation or Activation.by_name("linear")()
        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:

        if self._use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], -1)
            matrix_2 = torch.cat([matrix_2, bias2], -1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        return self._activation(final.squeeze(1) + self._bias)
