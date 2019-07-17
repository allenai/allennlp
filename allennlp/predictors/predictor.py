from typing import List, Iterator, Dict, Tuple, Any
import json
from contextlib import contextmanager
import numpy
from torch.utils.hooks import RemovableHandle
from torch import Tensor

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.dataset import Batch

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'atis_parser' : 'atis_parser',
        'basic_classifier': 'text_classifier',
        'biaffine_parser': 'biaffine-dependency-parser',
        'bidaf': 'machine-comprehension',
        'bidaf-ensemble': 'machine-comprehension',
        'bimpm': 'textual-entailment',
        'constituency_parser': 'constituency-parser',
        'coref': 'coreference-resolution',
        'crf_tagger': 'sentence-tagger',
        'decomposable_attention': 'textual-entailment',
        'dialog_qa': 'dialog_qa',
        'event2mind': 'event2mind',
        'naqanet': 'machine-comprehension',
        'simple_tagger': 'sentence-tagger',
        'srl': 'semantic-role-labeling',
        'srl_bert': 'semantic-role-labeling',
        'quarel_parser': 'quarel-parser',
        'wikitables_mml_parser': 'wikitables-parser'
}

class Predictor(Registrable):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        self._model = model
        self._dataset_reader = dataset_reader

    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return json.loads(line)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs) + "\n"

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def json_to_labeled_instances(self, inputs: JsonDict) -> List[Instance]:
        """
        Converts incoming json to a :class:`~allennlp.data.instance.Instance`,
        runs the model on the newly created instance, and adds labels to the
        :class:`~allennlp.data.instance.Instance`s given by the model's output.
        Returns
        -------
        List[instance]
        A list of :class:`~allennlp.data.instance.Instance`
        """
        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        new_instances = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    def get_gradients(self,
                      instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        Parameters
        ----------
        instances: List[Instance]

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
        The first item is a Dict of gradient entries for each input.
        The keys have the form  ``{grad_input_1: ..., grad_input_2: ... }``
        up to the number of inputs given. The second item is the model's output.

        Notes
        -----
        Takes a ``JsonDict`` representing the inputs of the model and converts
        them to :class:`~allennlp.data.instance.Instance`s, sends these through
        the model :func:`forward` function after registering hooks on the embedding
        layer of the model. Calls :func:`backward` on the loss and then removes the
        hooks.
        """
        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        outputs = self._model.decode(self._model.forward(**dataset.as_tensor_dict()))

        loss = outputs['loss']
        self._model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = 'grad_input_' + str(idx + 1)
            grad_dict[key] = grad.squeeze_(0).detach().cpu().numpy()

        return grad_dict, outputs

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        :class:`~allennlp.modules.text_field_embedder.basic_text_field_embbedder.BasicTextFieldEmbedder`
        class. Used to save the gradients of the embeddings for use in get_gradients()

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """
        def hook_layers(module, grad_in, grad_out): # pylint: disable=unused-argument
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        for module in self._model.modules():
            if isinstance(module, TextFieldEmbedder):
                backward_hooks.append(module.register_backward_hook(hook_layers))

        return backward_hooks

    @contextmanager
    def capture_model_internals(self) -> Iterator[dict]:
        """
        Context manager that captures the internal-module outputs of
        this predictor's model. The idea is that you could use it as follows:

        .. code-block:: python

            with predictor.capture_model_internals() as internals:
                outputs = predictor.predict_json(inputs)

            return {**outputs, "model_internals": internals}
        """
        results = {}
        hooks = []

        # First we'll register hooks to add the outputs of each module to the results dict.
        def add_output(idx: int):
            def _add_output(mod, _, outputs):
                results[idx] = {"name": str(mod), "output": sanitize(outputs)}
            return _add_output

        for idx, module in enumerate(self._model.modules()):
            if module != self._model:
                hook = module.register_forward_hook(add_output(idx))
                hooks.append(hook)

        # If you capture the return value of the context manager, you get the results dict.
        yield results

        # And then when you exit the context we remove all the hooks.
        for hook in hooks:
            hook.remove()

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        """
        This function takes a model's outputs for an Instance, and it labels that instance according
        to the output. For example, in classification this function labels the instance according
        to the class with the highest probability. This function is used to to compute gradients
        of what the model predicted. The return type is a list because in some tasks there are
        multiple predictions in the output (e.g., in NER a model predicts multiple spans). In this
        case, each instance in the returned list of Instances contains an individual
        entity prediction as the label.

        """
        # pylint: disable=unused-argument,no-self-use
        raise RuntimeError("implement this method for model interpretations or attacks")

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Converts a JSON object into an :class:`~allennlp.data.instance.Instance`
        and a ``JsonDict`` of information which the ``Predictor`` should pass through,
        such as tokenised inputs.
        """
        raise NotImplementedError

    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        return self.predict_batch_instance(instances)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    def _batch_json_to_instances(self, json_dicts: List[JsonDict]) -> List[Instance]:
        """
        Converts a list of JSON objects into a list of :class:`~allennlp.data.instance.Instance`s.
        By default, this expects that a "batch" consists of a list of JSON blobs which would
        individually be predicted by :func:`predict_json`. In order to use this method for
        batch prediction, :func:`_json_to_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        """
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    @classmethod
    def from_path(cls, archive_path: str, predictor_name: str = None, cuda_device: int = -1) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an archive path.

        If you need more detailed configuration options, such as overrides,
        please use `from_archive`.

        Parameters
        ----------
        archive_path: ``str``
            The path to the archive.
        predictor_name: ``str``, optional (default=None)
            Name that the predictor is registered as, or None to use the
            predictor associated with the model.
        cuda_device: ``int``, optional (default=-1)
            If `cuda_device` is >= 0, the model will be loaded onto the
            corresponding GPU. Otherwise it will be loaded onto the CPU.

        Returns
        -------
        A Predictor instance.
        """
        return Predictor.from_archive(load_archive(archive_path, cuda_device=cuda_device), predictor_name)

    @classmethod
    def from_archive(cls, archive: Archive, predictor_name: str = None) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, the default one for the model will be used.
        """
        # Duplicate the config so that the config inside the archive doesn't get consumed
        config = archive.config.duplicate()

        if not predictor_name:
            model_type = config.get("model").get("type")
            if not model_type in DEFAULT_PREDICTORS:
                raise ConfigurationError(f"No default predictor for model type {model_type}.\n"\
                                         f"Please specify a predictor explicitly.")
            predictor_name = DEFAULT_PREDICTORS[model_type]

        dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        model = archive.model
        model.eval()

        return Predictor.by_name(predictor_name)(model, dataset_reader)
