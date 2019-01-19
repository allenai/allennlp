# pylint: disable=no-self-use,invalid-name
from typing import Optional, Dict


import torch

from allennlp.nn.util import masked_softmax


class SelfAttentiveSentenceEncoder(torch.nn.Module):
    """
        The Self Attentive Sentence Encoder which is based on the paper https://arxiv.org/abs/1703.03130.
        This basically generates the encoded representation of the sentence by attending to different words
        in the sentence. The implementation also has an optional mask feature which normalizes the attention
        weights after assigning zeros to the padding tokens.

    Parameters
    ----------
    attention_size : ``int``
        This is the size of the intermediate attention layer which is similar to a linear layer. (d_a in paper)
    attention_heads : ``int``
        The number of different attention heads for which attention values need to be calculated. (r in paper)
    input_dim : ``int``
        The hidden dimensions of the tensor which will be input to the model.
    regularization_coeffecient : ``float``, optional (default = None)
        The coeffecient of the frobenius regularization term that needs to be added to the loss.
    """
    def __init__(self,
                 attention_size: int,
                 num_attention_heads: int,
                 input_dim: int,
                 regularization_coefficient: Optional[float] = None) -> None:

        super(SelfAttentiveSentenceEncoder, self).__init__()
        self._attention_size = attention_size
        self._num_attention_heads = num_attention_heads
        self._input_dim = input_dim
        self._regularization_coeffecient = regularization_coefficient

        self._linear_inner = torch.nn.Linear(input_dim, attention_size, bias=False)
        self._linear_outer = torch.nn.Linear(attention_size, num_attention_heads, bias=False)

    def init_weights(self) -> None:
        self.linear_first.weight.data.uniform_(-0.1, 0.1)
        self.linear_second.weight.data.uniform_(-0.1, 0.1)

    def forward(self, # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A dictionary of outputs containing the following items:
        representation : ``torch.FloatTensor``
            The final representation produced by attention of the shape (batch_size, input_dim).
        penalty : ``torch.FloatTensor``
            The frobenius norm based regularization penalty.
        attention : ``torch.FloatTensor``
            The values of attention corrosponding to the different attention heads having
            the shape (batch_size, num_attention_heads, time_steps).
        """
        # Shape (batch_size, timesteps, attention_size)
        attention_matrix = torch.tanh(self._linear_inner(inputs))
        # Shape (batch_size, timesteps, num_attention_heads)
        attention_vector = self._linear_outer(attention_matrix)
        # Shape (batch_size, timesteps, num_attention_heads)
        if mask is not None:
            mask = mask.unsqueeze(2)  # For unsqueezing mask to have three dimensions

        attention = masked_softmax(attention_vector, mask, dim=1)

        batch_size = inputs.shape[0]
        outputs = {"attention": attention}

        if self._regularization_coeffecient:
            outputs["regularization_loss"] = self.frobenius_regularization_penalty(attention)/batch_size

        # Shape (batch_size, num_attention_heads, input_dim)
        attended_representation = attention.transpose(1, 2)@inputs
        # Shape (batch_size, input_dim*num_attention_heads)
        outputs["representation"] = attended_representation.view(batch_size, -1)

        return outputs

    def frobenius_regularization_penalty(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            The matrix ||AAT - I|| for which the norm is to be calculated.

        Returns
        -------
        penalty : ``torch.Tensor``
            The norm calculated for a batch.
        """

        num_timesteps = attention.size(1)
        batch_size = attention.size(0)
        # Shape (batch_size, num_attention_heads, timesteps)
        attention_transpose = attention.transpose(1, 2)

        # Done to ensure that constant identity matrix is also created on the device model is running
        identity = torch.eye(num_timesteps, device=attention.device)

        # Shape (batch_size, timesteps, timesteps)
        identity = identity.unsqueeze(0).expand(batch_size, num_timesteps, num_timesteps)

        # Shape (batch_size, timesteps, timesteps)
        delta = attention @ attention_transpose - identity

        return torch.sum(torch.sum(torch.sum(delta ** 2, 1), 1) ** 0.5)
