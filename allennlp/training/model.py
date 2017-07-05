from typing import Dict
import torch


class Model(torch.nn.Module):
    """
    This abstract class represents a model to be trained. Rather than relying completely
    on the Pytorch Module, we modify the output spec of ``forward`` to be a dictionary.

    Models built using this API are still compatible with other pytorch models and can
    be used naturally as modules within other models - outputs are dictionaries, which
    can be unpacked and passed into other layers. One caveat to this is that if you
    wish to use an AllenNLP model inside a Container (such as nn.Sequential), you must
    interleave the models with a wrapper module which unpacks the dictionary into
    a list of tensors. TODO(Mark): Implement this.

    Finally, in order for your model to be trained using the :class:`~allennlp.training.Trainer`
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process.
    """

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        """
        Defines the forward pass of the model. In addition, to facilitate easy training,
        this method is designed to compute a loss function defined by a user.

        The input is comprised of everything required to perform a
        training update, `including` labels - you define the signature here!
        It is down to the user to ensure that inference can be performed
        without the presence of these labels. Hence, any inputs not available at
        inference time should only be used inside a conditional block.

        The intended sketch of this method is as follows:

        >>> def forward(self, input1, input2, targets=None):
        >>>     ....
        >>>     ....
        >>>     output1 = self.layer1(input1)
        >>>     output2 = self.layer2(input2)
        >>>     output_dict = {"output1": output1, "output2": output2}
        >>>     if targets is not None:
        >>>         # Function returning a scalar torch.Tensor, defined by the user.
        >>>         loss = self._compute_loss(output1, output2, targets)
        >>>         output_dict["loss"] = loss
        >>>
        >>>     return output_dict

        Parameters
        ----------
        inputs:
            Tensors comprising everything needed to perform a training update,
            `including` labels, which should be optional (i.e have a default value of None).
            At inference time, simply pass the relevant inputs, not including the labels.

        Returns
        -------
        output_dict: Dict[str, torch.Tensor]
            The outputs from the model. In order to train a model using the
            :class:`~allennlp.training.Trainer` api, you must provide a "loss"
            key pointing to a scalar torch.Tensor representing the loss to be optimized.
        """
        raise NotImplementedError
