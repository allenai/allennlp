from typing import List, Iterator, Dict, Tuple, Any, Type
import json
from contextlib import contextmanager

import numpy
from torch.utils.hooks import RemovableHandle
from torch import Tensor
from torch import backends

from allennlp.common import Registrable, plugins
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.batch import Batch
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.nn import util


class Predictor(Registrable):
    """
    a `Predictor` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        if frozen:
            model.eval()
        self._model = model
        self._dataset_reader = dataset_reader
        self.cuda_device = next(self._model.named_parameters())[1].get_device()

    def load_line(self, line: str) -> JsonDict:
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return json.loads(line)

    def dump_line(self, outputs: JsonDict) -> str:
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
        Converts incoming json to a [`Instance`](../data/instance.md),
        runs the model on the newly created instance, and adds labels to the
        `Instance`s given by the model's output.

        # Returns

        `List[instance]`
            A list of `Instance`'s.
        """

        instance = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        new_instances = self.predictions_to_labeled_instances(instance, outputs)
        return new_instances

    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        # Parameters

        instances : `List[Instance]`

        # Returns

        `Tuple[Dict[str, Any], Dict[str, Any]]`
            The first item is a Dict of gradient entries for each input.
            The keys have the form  `{grad_input_1: ..., grad_input_2: ... }`
            up to the number of inputs given. The second item is the model's output.

        # Notes

        Takes a `JsonDict` representing the inputs of the model and converts
        them to [`Instances`](../data/instance.md)), sends these through
        the model [`forward`](../models/model.md#forward) function after registering hooks on the embedding
        layer of the model. Calls `backward` on the loss and then removes the
        hooks.
        """
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self._model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        dataset_tensor_dict = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
        # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
        with backends.cudnn.flags(enabled=False):
            outputs = self._model.make_output_human_readable(
                self._model.forward(**dataset_tensor_dict)  # type: ignore
            )

            loss = outputs["loss"]
            self._model.zero_grad()
            loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # restore the original requires_grad values of the parameters
        for param_name, param in self._model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return grad_dict, outputs

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        [`BasicTextFieldEmbedder`](../modules/text_field_embedders/basic_text_field_embedder.md)
        class. Used to save the gradients of the embeddings for use in get_gradients()

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """

        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        embedding_layer = util.find_embedding_layer(self._model)
        backward_hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks

    @contextmanager
    def capture_model_internals(self) -> Iterator[dict]:
        """
        Context manager that captures the internal-module outputs of
        this predictor's model. The idea is that you could use it as follows:

        ```
            with predictor.capture_model_internals() as internals:
                outputs = predictor.predict_json(inputs)

            return {**outputs, "model_internals": internals}
        ```
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

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        """
        This function takes a model's outputs for an Instance, and it labels that instance according
        to the output. For example, in classification this function labels the instance according
        to the class with the highest probability. This function is used to to compute gradients
        of what the model predicted. The return type is a list because in some tasks there are
        multiple predictions in the output (e.g., in NER a model predicts multiple spans). In this
        case, each instance in the returned list of Instances contains an individual
        entity prediction as the label.
        """

        raise RuntimeError("implement this method for model interpretations or attacks")

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Converts a JSON object into an [`Instance`](../data/instance.md)
        and a `JsonDict` of information which the `Predictor` should pass through,
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
        Converts a list of JSON objects into a list of `Instance`s.
        By default, this expects that a "batch" consists of a list of JSON blobs which would
        individually be predicted by `predict_json`. In order to use this method for
        batch prediction, `_json_to_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        """
        instances = []
        for json_dict in json_dicts:
            instances.append(self._json_to_instance(json_dict))
        return instances

    @classmethod
    def from_path(
        cls,
        archive_path: str,
        predictor_name: str = None,
        cuda_device: int = -1,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
        import_plugins: bool = True,
    ) -> "Predictor":
        """
        Instantiate a `Predictor` from an archive path.

        If you need more detailed configuration options, such as overrides,
        please use `from_archive`.

        # Parameters

        archive_path : `str`
            The path to the archive.
        predictor_name : `str`, optional (default=`None`)
            Name that the predictor is registered as, or None to use the
            predictor associated with the model.
        cuda_device : `int`, optional (default=`-1`)
            If `cuda_device` is >= 0, the model will be loaded onto the
            corresponding GPU. Otherwise it will be loaded onto the CPU.
        dataset_reader_to_load : `str`, optional (default=`"validation"`)
            Which dataset reader to load from the archive, either "train" or
            "validation".
        frozen : `bool`, optional (default=`True`)
            If we should call `model.eval()` when building the predictor.
        import_plugins : `bool`, optional (default=`True`)
            If `True`, we attempt to import plugins before loading the predictor.
            This comes with additional overhead, but means you don't need to explicitly
            import the modules that your predictor depends on as long as those modules
            can be found by `allennlp.common.plugins.import_plugins()`.

        # Returns

        `Predictor`
            A Predictor instance.
        """
        if import_plugins:
            plugins.import_plugins()
        return Predictor.from_archive(
            load_archive(archive_path, cuda_device=cuda_device),
            predictor_name,
            dataset_reader_to_load=dataset_reader_to_load,
            frozen=frozen,
        )

    @classmethod
    def from_archive(
        cls,
        archive: Archive,
        predictor_name: str = None,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
    ) -> "Predictor":
        """
        Instantiate a `Predictor` from an [`Archive`](../models/archival.md);
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, we try to find a corresponding predictor in `DEFAULT_PREDICTORS`, or if
        one is not found, the base class (i.e. `Predictor`) will be used. Optionally specify
        which [`DatasetReader`](../data/dataset_readers/dataset_reader.md) should be loaded;
        otherwise, the validation one will be used if it exists followed by the training dataset reader.
        Optionally specify if the loaded model should be frozen, meaning `model.eval()` will be called.
        """
        # Duplicate the config so that the config inside the archive doesn't get consumed
        config = archive.config.duplicate()

        if not predictor_name:
            model_type = config.get("model").get("type")
            model_class, _ = Model.resolve_class_name(model_type)
            predictor_name = model_class.default_predictor
        predictor_class: Type[Predictor] = Predictor.by_name(  # type: ignore
            predictor_name
        ) if predictor_name is not None else cls

        if dataset_reader_to_load == "validation" and "validation_dataset_reader" in config:
            dataset_reader_params = config["validation_dataset_reader"]
        else:
            dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        model = archive.model
        if frozen:
            model.eval()

        return predictor_class(model, dataset_reader)
