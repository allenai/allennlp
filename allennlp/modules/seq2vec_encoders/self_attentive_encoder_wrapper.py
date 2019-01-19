
import torch

from allennlp.modules.self_attentive_sentence_encoder import SelfAttentiveSentenceEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

@Seq2VecEncoder.register('self-attentive-encoder')
class SelfAttentiveEncoderWrapper(Seq2VecEncoder):
    """
    The Self Attentive Encoder Wrapper is wrapper for the the modules/self_attentive_sentence_encoder.py.
    The encoder constructs a vector of representation for the sentence by attending to all time steps of
    sequence. The difference between the Wrapper and the module is that the wrapper just returns the
    vector representation whereas the module returns a loss, attention and the representation. If you
    intend to use the frobenius regularization loss and visualize the attention use the module directly
    rather then using the wrapper.

    Parameters
    ----------
    attention_size : ``int``
        This is the size of the intermediate attention layer which is similar
        to a linear layer. (d_a in paper)
    attention_heads : ``int``
        The number of different attention heads for which attention values need
        to be calculated. (r in paper)
    input_dim : ``int``
        The hidden dimensions of the tensor which will be input to the model.
    """
    def __init__(self,
                 attention_size: int,
                 num_attention_heads: int,
                 input_dim: int) -> None:
        super(SelfAttentiveEncoderWrapper, self).__init__()

        # Disabling regularization_loss calculation
        self._module = SelfAttentiveSentenceEncoder(attention_size, num_attention_heads,
                                                    input_dim, regularization_coefficient=None)
        self._input_dim = input_dim
        self._attention_size = attention_size
        self._num_attention_heads = num_attention_heads

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._num_attention_heads * self._input_dim

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        representation : ``torch.FloatTensor``
            The final representation produced by attention of the shape (batch_size, input_dim).
        """
        # Run the forward function
        output_dict = self._module(inputs, mask)

        return output_dict["representation"]
