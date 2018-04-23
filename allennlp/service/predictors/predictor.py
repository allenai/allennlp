from typing import Dict, List, Tuple
import json

from allennlp.common import Registrable
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'srl': 'semantic-role-labeling',
        'decomposable_attention': 'textual-entailment',
        'bidaf': 'machine-comprehension',
        'bidaf-ensemble': 'machine-comprehension',
        'simple_tagger': 'sentence-tagger',
        'crf_tagger': 'sentence-tagger',
        'coref': 'coreference-resolution',
        'constituency_parser': 'constituency-parser',
}

class Predictor(Registrable):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles making individual predictions
    that can be used for serving models through the web API, making predictions in bulk on the CLI, or
    making predictions programatically.
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

    def predict(self, **kwargs) -> JsonDict:
        raise NotImplementedError

    def predict_batch(self, inputs: List[JsonDict], cuda_device: int = -1):
        raise NotImplementedError

    def _default_predict_batch(self, built_instances: List[Tuple[Instance, Dict]], cuda_device: int = -1) \
            -> List[JsonDict]:
        instances, return_dicts = zip(*built_instances)
        outputs = self._model.forward_on_instances(instances, cuda_device)
        for output, return_dict in zip(outputs, return_dicts):
            return_dict.update(output)
        return sanitize(return_dicts)

    def _build_instances_batch(self, inputs: List[JsonDict]) -> List[Tuple[Instance, JsonDict]]:
        """
        Converts a list of input dictionaries into a list of :class:`~allennlp.data.instance.Instance`s.
        By default, this expects that a "batch" consists of a list of dictionaries which would
        individually be predicted by :func:`predict`. In order to use this method for
        batch prediction, :func:`_build_instance` should be implemented by the subclass, or
        if the instances have some dependency on each other, this method should be overridden
        directly.
        """

    @classmethod
    def from_path(cls, archive_path: str, predictor_name: str = None) -> 'Predictor':
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
        return Predictor.from_archive(load_archive(archive_path), predictor_name)

    @classmethod
    def from_archive(cls, archive: Archive, predictor_name: str = None) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, the default one for the model will be used.
        """
        config = archive.config

        if not predictor_name:
            model_type = config.get("model").get("type")
            assert model_type in DEFAULT_PREDICTORS, f"No default predictor for model type {model_type}.\n"\
                                                     f"Please specify a predictor explicitly."
            predictor_name = DEFAULT_PREDICTORS[model_type]

        dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        model = archive.model
        model.eval()

        return Predictor.by_name(predictor_name)(model, dataset_reader)


class DemoModel:
    """
    A demo model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)
