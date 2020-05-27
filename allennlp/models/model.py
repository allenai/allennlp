"""
`Model` is an abstract class representing
an AllenNLP model.
"""

import logging
import os
from typing import Dict, Union, List, Set, Type, Optional

try:
    from apex import amp
except ImportError:
    amp = None
import numpy
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.nn import util
from allennlp.nn.regularizers import RegularizerApplicator

logger = logging.getLogger(__name__)

# When training a model, many sets of weights are saved. By default we want to
# save/load this set of weights.
_DEFAULT_WEIGHTS = "best.th"


class Model(torch.nn.Module, Registrable):
    """
    This abstract class represents a model to be trained. Rather than relying completely
    on the Pytorch Module, we modify the output spec of `forward` to be a dictionary.

    Models built using this API are still compatible with other pytorch models and can
    be used naturally as modules within other models - outputs are dictionaries, which
    can be unpacked and passed into other layers. One caveat to this is that if you
    wish to use an AllenNLP model inside a Container (such as nn.Sequential), you must
    interleave the models with a wrapper module which unpacks the dictionary into
    a list of tensors.

    In order for your model to be trained using the [`Trainer`](../training/trainer.md)
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process.

    Finally, you can optionally implement :func:`Model.get_metrics` in order to make use
    of early stopping and best-model serialization based on a validation metric in
    `Trainer`. Metrics that begin with "_" will not be logged
    to the progress bar by `Trainer`.

    The `from_archive` method on this class is registered as a `Model` with name "from_archive".
    So, if you are using a configuration file, you can specify a model as `{"type": "from_archive",
    "archive_file": "/path/to/archive.tar.gz"}`, which will pull out the model from the given
    location and return it.

    # Parameters

    vocab: `Vocabulary`
        There are two typical use-cases for the `Vocabulary` in a `Model`: getting vocabulary sizes
        when constructing embedding matrices or output classifiers (as the vocabulary holds the
        number of classes in your output, also), and translating model output into human-readable
        form.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "model", it gets specified as a top-level parameter, then is passed in to the model
        separately.
    regularizer: `RegularizerApplicator`, optional
        If given, the `Trainer` will use this to regularize model parameters.
    """

    _warn_for_unseparable_batches: Set[str] = set()
    default_predictor: Optional[str] = None

    def __init__(self, vocab: Vocabulary, regularizer: RegularizerApplicator = None) -> None:
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

    def get_parameters_for_histogram_tensorboard_logging(self) -> List[str]:
        """
        Returns the name of model parameters used for logging histograms to tensorboard.
        """
        return [name for name, _ in self.named_parameters()]

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
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

        # Parameters

        *inputs : `Any`
            Tensors comprising everything needed to perform a training update, `including` labels,
            which should be optional (i.e have a default value of `None`).  At inference time,
            simply pass the relevant inputs, not including the labels.

        # Returns

        output_dict : `Dict[str, torch.Tensor]`
            The outputs from the model. In order to train a model using the
            `Trainer` api, you must provide a "loss" key pointing to a
            scalar `torch.Tensor` representing the loss to be optimized.
        """
        raise NotImplementedError

    def forward_on_instance(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Takes an [`Instance`](../data/instance.md), which typically has raw text in it, converts
        that text into arrays using this model's [`Vocabulary`](../data/vocabulary.md), passes those
        arrays through `self.forward()` and `self.make_output_human_readable()` (which by default
        does nothing) and returns the result.  Before returning the result, we convert any
        `torch.Tensors` into numpy arrays and remove the batch dimension.
        """
        return self.forward_on_instances([instance])[0]

    def forward_on_instances(self, instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
        """
        Takes a list of `Instances`, converts that text into arrays using this model's `Vocabulary`,
        passes those arrays through `self.forward()` and `self.make_output_human_readable()` (which
        by default does nothing) and returns the result.  Before returning the result, we convert
        any `torch.Tensors` into numpy arrays and separate the batched output into a list of
        individual dicts per instance. Note that typically this will be faster on a GPU (and
        conditionally, on a CPU) than repeated calls to `forward_on_instance`.

        # Parameters

        instances : `List[Instance]`, required
            The instances to run the model on.

        # Returns

        A list of the models output for each instance.
        """
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.make_output_human_readable(self(**model_input))

            instance_separated_output: List[Dict[str, numpy.ndarray]] = [
                {} for _ in dataset.instances
            ]
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
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Takes the result of `forward` and makes it human readable.  Most of the time, the only thing
        this method does is convert tokens / predicted labels from tensors to strings that humans
        might actually understand.  Somtimes you'll also do an argmax or something in here, too, but
        that most often happens in `Model.forward`, before you compute your metrics.

        This method `modifies` the input dictionary, and also `returns` the same dictionary.

        By default in the base class we do nothing.
        """

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Returns a dictionary of metrics. This method will be called by
        `allennlp.training.Trainer` in order to compute and use model metrics for early
        stopping and model serialization.  We return an empty dictionary here rather than raising
        as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
        passed, as frequently a metric accumulator will have some state which should be reset
        between epochs. This is also compatible with [`Metric`s](../training/metrics/metric.md). Metrics
        should be populated during the call to `forward`, with the `Metric` handling the accumulation of
        the metric until this method is called.
        """

        return {}

    def _get_prediction_device(self) -> int:
        """
        This method checks the device of the model parameters to determine the cuda_device
        this model should be run on for predictions.  If there are no parameters, it returns -1.

        # Returns

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
        by a class attribute `_warn_for_unseperable_batches` because it would be extremely verbose
        otherwise.
        """
        if output_key not in self._warn_for_unseparable_batches:
            logger.warning(
                f"Encountered the {output_key} key in the model's return dictionary which "
                "couldn't be split by the batch size. Key will be ignored."
            )
            # We only want to warn once for this key,
            # so we set this to false so we don't warn again.
            self._warn_for_unseparable_batches.add(output_key)

    @classmethod
    def _load(
        cls,
        config: Params,
        serialization_dir: str,
        weights_file: Optional[str] = None,
        cuda_device: int = -1,
        opt_level: Optional[str] = None,
    ) -> "Model":
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        model_params = config.get("model")

        training_params = config.get("trainer", Params({}))
        opt_level = opt_level or training_params.get("opt_level")

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings from.  We're now _loading_ the model, so those embeddings will already be
        # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
        # want the code to look for it, so we remove it from the parameters here.
        remove_pretrained_embedding_params(model_params)
        model = Model.from_params(vocab=vocab, params=model_params)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        # If opt_level is not None (i.e. it exists in the loaded models params or was provided
        # as argument to this method), call amp.initialize on the loaded model.
        # Log a warning if amp is not installed or we are loading onto the cpu so that these
        # cases do not pass silently.
        if opt_level is not None:
            if amp is None:
                logger.warning(
                    (
                        f"Apex must be installed to enable mixed-precision via amp."
                        f" Got opt_level is not None (opt_level={opt_level}) but Apex is not installed."
                        " Any further training or inference will happen at full-precision."
                    )
                )
            if cuda_device == -1:
                logger.warning(
                    (
                        f"A CUDA device must be specified to enable mixed-precision via amp."
                        f" Got cuda_device=={cuda_device} but opt_level is not None (opt_level={opt_level})."
                        " Any further training or inference will happen at full-precision."
                    )
                )
            if amp is not None and cuda_device >= 0:
                model = amp.initialize(model, opt_level=opt_level)

        # If vocab+embedding extension was done, the model initialized from from_params
        # and one defined by state dict in weights_file might not have same embedding shapes.
        # Eg. when model embedder module was transferred along with vocab extension, the
        # initialized embedding weight shape would be smaller than one in the state_dict.
        # So calling model embedding extension is required before load_state_dict.
        # If vocab and model embeddings are in sync, following would be just a no-op.
        model.extend_embedder_vocab()

        model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))
        model.load_state_dict(model_state)

        return model

    @classmethod
    def load(
        cls,
        config: Params,
        serialization_dir: str,
        weights_file: Optional[str] = None,
        cuda_device: int = -1,
        opt_level: Optional[str] = None,
    ) -> "Model":
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.

        # Parameters

        config : `Params`
            The configuration that was used to train the model. It should definitely
            have a `model` section, and should probably have a `trainer` section
            as well.
        serialization_dir: `str = None`
            The directory containing the serialized weights, parameters, and vocabulary
            of the model.
        weights_file: `str = None`
            By default we load the weights from `best.th` in the serialization
            directory, but you can override that value here.
        cuda_device: `int = -1`
            By default we load the model on the CPU, but if you want to load it
            for GPU usage you can specify the id of your GPU here
        opt_level : `str`, optional (default = `None`)
            Each `opt_level` establishes a set of properties that govern Amp’s implementation of pure or mixed
            precision training. Must be a choice of `"O0"`, `"O1"`, `"O2"`, or `"O3"`.
            See the Apex [documentation](https://nvidia.github.io/apex/amp.html#opt-levels-and-properties) for
            more details. If `None`, defaults to the `opt_level` found in the model params. If `cuda_device==-1`,
            Amp is not used and this argument is ignored.

        # Returns

        model : `Model`
            The model specified in the configuration, loaded with the serialized
            vocabulary and the trained weights.
        """

        # Peak at the class of the model.
        model_type = (
            config["model"] if isinstance(config["model"], str) else config["model"]["type"]
        )

        # Load using an overridable _load method.
        # This allows subclasses of Model to override _load.

        model_class: Type[Model] = cls.by_name(model_type)  # type: ignore
        if not isinstance(model_class, type):
            # If you're using from_archive to specify your model (e.g., for fine tuning), then you
            # can't currently override the behavior of _load; we just use the default Model._load.
            # If we really need to change this, we would need to implement a recursive
            # get_model_class method, that recurses whenever it finds a from_archive model type.
            model_class = Model
        return model_class._load(config, serialization_dir, weights_file, cuda_device, opt_level)

    def extend_embedder_vocab(self, embedding_sources_mapping: Dict[str, str] = None) -> None:
        """
        Iterates through all embedding modules in the model and assures it can embed
        with the extended vocab. This is required in fine-tuning or transfer learning
        scenarios where model was trained with original vocabulary but during
        fine-tuning/transfer-learning, it will have it work with extended vocabulary
        (original + new-data vocabulary).

        # Parameters

        embedding_sources_mapping : `Dict[str, str]`, optional (default = `None`)
            Mapping from model_path to pretrained-file path of the embedding
            modules. If pretrained-file used at time of embedding initialization
            isn't available now, user should pass this mapping. Model path is
            path traversing the model attributes upto this embedding module.
            Eg. "_text_field_embedder.token_embedder_tokens".
        """
        # self.named_modules() gives all sub-modules (including nested children)
        # The path nesting is already separated by ".": eg. parent_module_name.child_module_name
        embedding_sources_mapping = embedding_sources_mapping or {}
        for model_path, module in self.named_modules():
            if hasattr(module, "extend_vocab"):
                pretrained_file = embedding_sources_mapping.get(model_path)
                module.extend_vocab(
                    self.vocab, extension_pretrained_file=pretrained_file, model_path=model_path,
                )

    @classmethod
    def from_archive(cls, archive_file: str, vocab: Vocabulary = None) -> "Model":
        """
        Loads a model from an archive file.  This basically just calls
        `return archival.load_archive(archive_file).model`.  It exists as a method here for
        convenience, and so that we can register it for easy use for fine tuning an existing model
        from a config file.

        If `vocab` is given, we will extend the loaded model's vocabulary using the passed vocab
        object (including calling `extend_embedder_vocab`, which extends embedding layers).
        """
        from allennlp.models.archival import load_archive  # here to avoid circular imports

        model = load_archive(archive_file).model
        if vocab:
            model.vocab.extend_from_vocab(vocab)
            model.extend_embedder_vocab()
        return model


# We can't decorate `Model` with `Model.register()`, because `Model` hasn't been defined yet.  So we
# put this down here.
Model.register("from_archive", constructor="from_archive")(Model)


def remove_pretrained_embedding_params(params: Params):
    if isinstance(params, Params):  # The model could possibly be a string, for example.
        keys = params.keys()
        if "pretrained_file" in keys:
            del params["pretrained_file"]
        for value in params.values():
            if isinstance(value, Params):
                remove_pretrained_embedding_params(value)
