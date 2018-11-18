"""
:py:class:`Model` is an abstract class representing
an AllenNLP model.
"""

import logging
import os
from typing import Dict, Union, List, Set

import numpy
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.nn.regularizers import RegularizerApplicator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# When training a model, many sets of weights are saved. By default we want to
# save/load this set of weights.
_DEFAULT_WEIGHTS = "best.th"


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
    :class:`~allennlp.training.Trainer`. Metrics that begin with "_" will not be logged
    to the progress bar by :class:`~allennlp.training.Trainer`.
    """
    _warn_for_unseparable_batches: Set[str] = set()

    def __init__(self,
                 vocab: Vocabulary,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__()
        self.vocab = vocab
        self._regularizer = regularizer

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:
        """
        Computes the regularization penalty for the model.
        Returns 0 if the model was not configured to use regularization.
        """
        if self._regularizer is None:
            return 0.0
        else:
            return self._regularizer(self)

    def get_parameters_for_histogram_tensorboard_logging( # pylint: disable=invalid-name
            self) -> List[str]:
        """
        Returns the name of model parameters used for logging histograms to tensorboard.
        """
        return [name for name, _ in self.named_parameters()]

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        """
        Defines the forward pass of the model. In addition, to facilitate easy training,
        this method is designed to compute a loss function defined by a user.

        The input is comprised of everything required to perform a
        training update, `including` labels - you define the signature here!
        It is down to the user to ensure that inference can be performed
        without the presence of these labels. Hence, any inputs not available at
        inference time should only be used inside a conditional block.

        The intended sketch of this method is as follows::

            def forward(self, input1, input2, targets=None):
                ....
                ....
                output1 = self.layer1(input1)
                output2 = self.layer2(input2)
                output_dict = {"output1": output1, "output2": output2}
                if targets is not None:
                    # Function returning a scalar torch.Tensor, defined by the user.
                    loss = self._compute_loss(output1, output2, targets)
                    output_dict["loss"] = loss
                return output_dict

        Parameters
        ----------
        inputs:
            Tensors comprising everything needed to perform a training update, `including` labels,
            which should be optional (i.e have a default value of ``None``).  At inference time,
            simply pass the relevant inputs, not including the labels.

        Returns
        -------
        output_dict: ``Dict[str, torch.Tensor]``
            The outputs from the model. In order to train a model using the
            :class:`~allennlp.training.Trainer` api, you must provide a "loss" key pointing to a
            scalar ``torch.Tensor`` representing the loss to be optimized.
        """
        raise NotImplementedError

    def forward_on_instance(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Takes an :class:`~allennlp.data.instance.Instance`, which typically has raw text in it,
        converts that text into arrays using this model's :class:`Vocabulary`, passes those arrays
        through :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and remove the batch dimension.
        """
        return self.forward_on_instances([instance])[0]

    def forward_on_instances(self,
                             instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
        """
        Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
        arrays using this model's :class:`Vocabulary`, passes those arrays through
        :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and separate the
        batched output into a list of individual dicts per instance. Note that typically
        this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
        :func:`forward_on_instance`.

        Parameters
        ----------
        instances : List[Instance], required
            The instances to run the model on.
        cuda_device : int, required
            The GPU device to use.  -1 means use the CPU.

        Returns
        -------
        A list of the models output for each instance.
        """
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.decode(self(**model_input))

            instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                outputs[name] = output
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Takes the result of :func:`forward` and runs inference / decoding / whatever
        post-processing you need to do your model.  The intent is that ``model.forward()`` should
        produce potentials or probabilities, and then ``model.decode()`` can take those results and
        run some kind of beam search or constrained inference or whatever is necessary.  This does
        not handle all possible decoding use cases, but it at least handles simple kinds of
        decoding.

        This method `modifies` the input dictionary, and also `returns` the same dictionary.

        By default in the base class we do nothing.  If your model has some special decoding step,
        override this method.
        """
        # pylint: disable=no-self-use
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Returns a dictionary of metrics. This method will be called by
        :class:`allennlp.training.Trainer` in order to compute and use model metrics for early
        stopping and model serialization.  We return an empty dictionary here rather than raising
        as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
        passed, as frequently a metric accumulator will have some state which should be reset
        between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
        should be populated during the call to ``forward``, with the
        :class:`~allennlp.training.Metric` handling the accumulation of the metric until this
        method is called.
        """
        # pylint: disable=unused-argument,no-self-use
        return {}

    def _get_prediction_device(self) -> int:
        """
        This method checks the device of the model parameters to determine the cuda_device
        this model should be run on for predictions.  If there are no parameters, it returns -1.

        Returns
        -------
        The cuda device this model should run on for predictions.
        """
        devices = {util.get_device_of(param) for param in self.parameters()}

        if len(devices) > 1:
            devices_string = ", ".join(str(x) for x in devices)
            raise ConfigurationError(f"Parameters have mismatching cuda_devices: {devices_string}")
        elif len(devices) == 1:
            return devices.pop()
        else:
            return -1

    def _maybe_warn_for_unseparable_batches(self, output_key: str):
        """
        This method warns once if a user implements a model which returns a dictionary with
        values which we are unable to split back up into elements of the batch. This is controlled
        by a class attribute ``_warn_for_unseperable_batches`` because it would be extremely verbose
        otherwise.
        """
        if  output_key not in self._warn_for_unseparable_batches:
            logger.warning(f"Encountered the {output_key} key in the model's return dictionary which "
                           "couldn't be split by the batch size. Key will be ignored.")
            # We only want to warn once for this key,
            # so we set this to false so we don't warn again.
            self._warn_for_unseparable_batches.add(output_key)

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1) -> 'Model':
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_dir, 'vocabulary')
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
        vocab = Vocabulary.by_name(vocab_choice).from_files(vocab_dir)

        model_params = config.get('model')

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings from.  We're now _loading_ the model, so those embeddings will already be
        # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
        # want the code to look for it, so we remove it from the parameters here.
        remove_pretrained_embedding_params(model_params)
        model = Model.from_params(vocab=vocab, params=model_params)
        model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))
        model.load_state_dict(model_state)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        return model

    @classmethod
    def load(cls,
             config: Params,
             serialization_dir: str,
             weights_file: str = None,
             cuda_device: int = -1) -> 'Model':
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.

        Parameters
        ----------
        config: Params
            The configuration that was used to train the model. It should definitely
            have a `model` section, and should probably have a `trainer` section
            as well.
        serialization_dir: str = None
            The directory containing the serialized weights, parameters, and vocabulary
            of the model.
        weights_file: str = None
            By default we load the weights from `best.th` in the serialization
            directory, but you can override that value here.
        cuda_device: int = -1
            By default we load the model on the CPU, but if you want to load it
            for GPU usage you can specify the id of your GPU here


        Returns
        -------
        model: Model
            The model specified in the configuration, loaded with the serialized
            vocabulary and the trained weights.
        """

        # Peak at the class of the model.
        model_type = config["model"]["type"]

        # Load using an overridable _load method.
        # This allows subclasses of Model to override _load.
        # pylint: disable=protected-access
        return cls.by_name(model_type)._load(config, serialization_dir, weights_file, cuda_device)


def remove_pretrained_embedding_params(params: Params):
    keys = params.keys()
    if 'pretrained_file' in keys:
        del params['pretrained_file']
    for value in params.values():
        if isinstance(value, Params):
            remove_pretrained_embedding_params(value)
