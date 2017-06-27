
from typing import List, Dict, Union
import torch


class Model(torch.nn.Module):

    """
    This abstract class represents a model to be trained. Rather than relying on the
    raw torch Module, we provide a slightly different API.


    """

    def forward(self, inputs):

        raise NotImplementedError


    def compute_loss(self,
                     model_predictions: Union[torch.Tensor, List[torch.Tensor]],
                     targets: Union[torch.Tensor, List[torch.Tensor]],
                     model_state: Dgit ict[]):