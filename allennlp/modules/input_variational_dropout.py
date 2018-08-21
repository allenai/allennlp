import torch

class InputVariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" (https://arxiv.org/abs/1506.02142) to a
    3D tensor.

    This module accepts a 3D tensor of shape ``(batch_size, num_timesteps, embedding_dim)``
    and samples a single dropout mask of shape ``(batch_size, embedding_dim)`` and applies
    it to every time step.
    """
    def forward(self, input_tensor):
        # pylint: disable=arguments-differ
        """
        Apply dropout to input tensor.

        Parameters
        ----------
        input_tensor: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)``

        Returns
        -------
        output: ``torch.FloatTensor``
            A tensor of shape ``(batch_size, num_timesteps, embedding_dim)`` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor
