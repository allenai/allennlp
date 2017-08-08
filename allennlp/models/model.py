from typing import Dict
import torch

from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.data import Vocabulary


class Model(torch.nn.Module, Registrable):
    """
    This abstract class represents a model to be trained. Rather than relying completely
    on the Pytorch Module, we modify the output spec of ``forward`` to be a dictionary.

    Models built using this API are still compatible with other pytorch models and can
    be used naturally as modules within other models - outputs are dictionaries, which
    can be unpacked and passed into other layers. One caveat to this is that if you
    wish to use an AllenNLP model inside a Container (such as nn.Sequential), you must
    interleave the models with a wrapper module which unpacks the dictionary into
    a list of tensors.

    Finally, the output dictionary of your model can contain 2 special keys which we
    use if you train a model using the :class:`~allennlp.training.Trainer`. These are:

    loss : ``torch.Variable``, required.
        The loss to be optimized. Required for a model to be trained using ``Trainer``.

    has_metrics : bool, optional.
        If your model has this key in it's output dictionary, :func:`Model.get_metrics`
        will be called in the training loop after each epoch. This allows you to implement
        metrics which accumulate as your model trains.

    in order for your model to be trained using the :class:`~allennlp.training.Trainer`
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process. Additionally, it may include a
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

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Returns a dictionary of metrics. If the ``forward`` call of your model returns a
        ``has_metrics`` key, this will be called by :class:`allennlp.training.Trainer`
        in order to compute and use model metrics for early stopping and model serialisation.
        """
        pass

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        choice = params.pop_choice("type", cls.list_available())
        return cls.by_name(choice).from_params(vocab, params)
