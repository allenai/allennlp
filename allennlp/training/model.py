from typing import Any, List, Dict, Optional, Union
import torch


class Model(torch.nn.Module):
    """
    This abstract class represents a model to be trained. Rather than relying on the
    raw torch Module, we provide a slightly different API, particularly to align with
    data processing code for textual data. As textual data pre-processing and deep NLP
    model inputs tend to be more complicated than for other modalities, we use
    dictionaries of tensors to pass inputs to the model, to keep on top of what's
    going on. It also makes reading model implementations easier to connect to
    data generation, because model inputs are named.
    """

    def forward(self,  # pylint: disable=arguments-differ
                inputs: Dict[str, torch.Tensor],
                compute_loss: bool = False) -> (Union[torch.Tensor, List[torch.Tensor]], Optional[float]):
        """
        Defines the forward pass of the model. In addition, to facilitate easy training,
        this method is designed to compute a loss function defined by a user.

        The input dictionary is comprised of everything required to perform a
        training update, `including` labels. It is down to the user to ensure
        that inference can be performed (using compute_loss = False) without the
        presence of these labels! Hence, any inputs not available at inference
        time should only be used inside an ``if compute_loss:`` conditional block.

        The intended sketch of this method is as follows:

        >>> def forward(inputs, compute_loss=False):
        >>>     ....
        >>>     ....
        >>>     outputs = self.last_layer(inputs)
        >>>
        >>>     if compute_loss:
        >>>         loss = self.compute_loss(outputs, targets, model_state)
        >>>         return outputs, loss
        >>>     else:
        >>>         return outputs, None

        Parameters
        ----------
        inputs: Dict[str, torch.Tensor], required.
            A dictionary of tensors comprising everything needed to perform a
            training update, `including` labels. At inference time, simply
            pass an input dictionary which does not include labels and
        compute_loss: bool, optional (default = False)
            If true, this method should call :func:`compute_loss` and return
            both the model outputs and a scalar loss to optimise.

        Returns
        -------

        outputs: torch.Tensor or List[torch.Tensor]
            The outputs from the model.
        loss: float, optional
            If ``compute_outputs=True``, return a scalar loss to optimise.

        """

        raise NotImplementedError

    def compute_loss(self,
                     model_output: Union[torch.Tensor, List[torch.Tensor]],
                     targets: Union[torch.Tensor, List[torch.Tensor]],
                     model_state: Dict[str, Any] = None) -> float:
        """
        Computes a scalar loss function to optimise. This method is designed
        to be called and returned by :func:`forward` when `compute_loss = True`.
        It is part of the public interface however, as there are scenarios in which
        direct evaluation may be useful, such as during model comparison.

        Parameters
        ----------
        model_output : torch.Tensor or List[torch.Tensor], required.
            The output of the result of calling :func:`forward`. This
            can be a list, as complex models may have many outputs.
        targets : torch.Tensor or List[torch.Tensor], required.
            The gold targets for computing some loss function with respect to
            the model outputs.
        model_state : Dict[str, Any], optional (default = None)
            As Pytorch creates dynamic computation graphs, it is possible for your loss
            function to change depending on some model state. At it's simplest, this
            might be your original inputs, if you were training on multi-modal
            data where not every training instance has every mode, for example.
            This dictionary is where model state required for computing the loss
            should be passed.

        Returns
        -------
        A scalar loss to be optimised.
        """
        raise NotImplementedError
