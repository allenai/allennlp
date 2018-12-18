from typing import List
import json

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.introspection import store_function_results
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'atis_parser' : 'atis_parser',
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
        'simple_tagger': 'sentence-tagger',
        'srl': 'semantic-role-labeling',
        'quarel_parser': 'quarel-parser',
        'wikitables_mml_parser': 'wikitables-parser'
}

class Predictor(Registrable):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 return_model_internals: bool = False) -> None:
        self._model = model
        self._dataset_reader = dataset_reader
        self._return_model_internals = return_model_internals

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

    def predict_instance(self, instance: Instance) -> JsonDict:
        internal_module_results = {}
        hooks = []

        if self._return_model_internals:
            store_function_results(True)

            def add_output(idx: int):
                def _add_output(mod, _, outputs):
                    internal_module_results[idx] = {"name": str(mod), "output": outputs}
                return _add_output

            hooks = [module.register_forward_hook(add_output(i))
                     for i, module in enumerate(self._model.modules())
                     if module != self._model]

        outputs = self._model.forward_on_instance(instance)

        if self._return_model_internals:
            # Collect results of modules
            if internal_module_results:
                outputs['_internal_module_results'] = internal_module_results

            # Collect results of functions
            internal_function_results = getattr(self._model, '_stored_function_results', [])
            if internal_function_results:
                outputs['_internal_function_results'] = internal_function_results[:]

            # And clean up
            internal_function_results.clear()
            store_function_results(False)

            for hook in hooks:
                hook.remove()

        return sanitize(outputs)

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
    def from_path(cls,
                  archive_path: str,
                  predictor_name: str = None,
                  return_model_internals: bool = False) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an archive path.

        If you need more detailed configuration options, such as running the predictor on the GPU,
        please use `from_archive`.

        Parameters
        ----------
        archive_path The path to the archive.

        Returns
        -------
        A Predictor instance.
        """
        return Predictor.from_archive(load_archive(archive_path), predictor_name, return_model_internals)

    @classmethod
    def from_archive(cls,
                     archive: Archive,
                     predictor_name: str = None,
                     return_model_internals: bool = False) -> 'Predictor':
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

        return Predictor.by_name(predictor_name)(model, dataset_reader, return_model_internals)
