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

    In order for your model to be trained using the :class:`~allennlp.training.Trainer`
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process.

    Finally, you can optionally implement :func:`Model.get_metrics` in order to make use
    of early stopping and best-model serialization based on a validation metric in
    :class:`~allennlp.training.Trainer`.
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
        Returns a dictionary of metrics. This method will be called by
        :class:`allennlp.training.Trainer` in order to compute and use model metrics for early
        stopping and model serialisation.  We return an empty dictionary here rather than raising
        as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
        passed, as frequently a metric accumulator will have some state which should be reset
        between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
        should be populated during the call to ``forward``, with the
        :class:`~allennlp.training.Metric`s handling the accumulation of the metric until this
        method is called.
        """
        return {}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params):
        choice = params.pop_choice("type", cls.list_available())
        return cls.by_name(choice).from_params(vocab, params)
