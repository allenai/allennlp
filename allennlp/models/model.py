from typing import Dict
import tarfile
import tempfile
import os
import logging
import shutil

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.data import Vocabulary
from allennlp.nn.util import device_mapping

import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# We archive a model by creating a tar.gz file with its weights, config, and vocabulary.
# These are the *known names* under which we archive the config and weights.
_CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"

# When training a model, many sets of weights are saved. By default we want to
# archive this set of weights.
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

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Returns a dictionary of metrics. This method will be called by
        :class:`allennlp.training.Trainer` in order to compute and use model metrics for early
        stopping and model serialisation.  We return an empty dictionary here rather than raising
        as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
        passed, as frequently a metric accumulator will have some state which should be reset
        between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
        should be populated during the call to ``forward``, with the
        :class:`~allennlp.training.Metric` handling the accumulation of the metric until this
        method is called.
        """
        # pylint: disable=unused-argument,no-self-use
        return {}

    def archive(self,                                       # pylint: disable=no-self-use
                serialization_prefix: str,
                config_file: str,
                weights: str = _DEFAULT_WEIGHTS) -> None:
        """
        Archives the model weights, its training configuration, and its
        vocabulary to `model.tar.gz`

        Parameters
        ----------
        serialization_prefix: ``str``
            The directory where the weights and vocabulary are written out.
        config_file: ``str``
            The path to the experiment configuration file used to train the model.
        weights: ``str``, optional (default=_DEFAULT_WEIGHTS)
            Which weights file to include in the archive. The default is ``best.th``.
        """
        archive_file = os.path.join(serialization_prefix, "model.tar.gz")
        logger.info("archiving weights and vocabulary to %s", archive_file)
        with tarfile.open(archive_file, 'w:gz') as archive:
            archive.add(config_file, arcname=_CONFIG_NAME)
            archive.add(os.path.join(serialization_prefix, weights),
                        arcname=_WEIGHTS_NAME)
            archive.add(os.path.join(serialization_prefix, "vocabulary"),
                        arcname="vocabulary")

    @classmethod
    def from_archive(cls,
                     archive_file: str,
                     cuda_device: int = -1) -> 'Model':
        """
        Instantiates a model from an archived `tar.gz` file.

        Parameters
        ----------
        archive_file: ``str``
            The archive file to load the model from.
        cuda_device: ``int``, optional (default = -1)
            If `cuda_device` is >= 0, the model will be loaded onto the
            corresponding GPU. Otherwise it will be loaded to CPU.
        """
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        logger.info("extracting archive file %s to temp dir %s", archive_file, tempdir)
        with tarfile.open(archive_file, 'r:gz') as archive:
            archive.extractall(tempdir)

        # Load config
        config = Params.from_file(os.path.join(tempdir, _CONFIG_NAME))

        # Instantiate model
        model = Model.load(config,
                           weights_file=os.path.join(tempdir, _WEIGHTS_NAME),
                           serialization_prefix=tempdir,
                           cuda_device=cuda_device)

        # Clean up temp dir
        shutil.rmtree(tempdir)

        return model


    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Model':
        choice = params.pop_choice("type", cls.list_available())
        return cls.by_name(choice).from_params(vocab, params)

    @classmethod
    def load(cls,
             config: Params,
             serialization_prefix: str = None,
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
        serialization_prefix: str = None
            By default we look at `config['trainer']['serialization_prefix']` to
            get the path to the serialized model, but you can override that
            value here.
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
        trainer_config = config.get("trainer", {})
        serialization_prefix = (serialization_prefix or
                                trainer_config.get('serialization_prefix'))
        if serialization_prefix is None:
            raise ConfigurationError('serialization_prefix must be specified')

        weights_file = weights_file or os.path.join(serialization_prefix, _DEFAULT_WEIGHTS)

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_prefix, 'vocabulary')
        vocab = Vocabulary.from_files(vocab_dir)

        model = Model.from_params(vocab, config.get('model'))
        model_state = torch.load(weights_file, map_location=device_mapping(cuda_device))
        model.load_state_dict(model_state)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        return model
