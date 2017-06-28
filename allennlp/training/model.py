
from typing import Any, List, Dict, Union
import torch


class Model(torch.nn.Module):

    """
    This abstract class represents a model to be trained. Rather than relying on the
    raw torch Module, we provide a slightly different API.
    """

    def forward(self, inputs: Dict[str, torch.Tensor], compute_loss: bool = False):
        """
        Defines the forward pass of the model. In addition, to facilitate easy training,
        this method is designed to compute a loss function defined by a user.

        The intended sketch of this function is as follows:

        >>> def forward(inputs, compute_loss=False):
        >>>     ....
        >>>     ....
        >>>     outputs = self.last_layer(inputs)
        >>>
        >>>     if compute_loss:
        >>>         return outputs, self.compute_loss(outputs, targets, model_state)
        >>>     else:
        >>>     return outputs

        Parameters
        ----------

        inputs: Dict[str, torch.Tensor], required.
            A dictionary of tensors comprising everything needed to



        """

        raise NotImplementedError

    def compute_loss(self,
                     model_output: Union[torch.Tensor, List[torch.Tensor]],
                     targets: Union[torch.Tensor, List[torch.Tensor]],
                     model_state: Dict[str, Any]) -> float:
        """

        Computes a scalar loss function to optimise. This method is designed
        to be called and returned only by the forward

        """
        raise NotImplementedError
