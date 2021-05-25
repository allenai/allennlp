import collections
from typing import Type, List, Dict

from overrides import overrides

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.models.multitask import MultiTaskModel
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import sanitize
from allennlp.data.fields import MetadataField
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers import MultiTaskDatasetReader


@Predictor.register("multitask")
class MultiTaskPredictor(Predictor):
    """
    Predictor for multitask models.

    Registered as a `Predictor` with name "multitask".

    This predictor is tightly coupled to `MultiTaskDatasetReader` and `MultiTaskModel`, and will not work if
    used with other readers or models.
    """

    _WRONG_READER_ERROR = (
        "MultitaskPredictor is designed to work with MultiTaskDatasetReader. "
        + "If you have a different DatasetReader, you have to write your own "
        + "Predictor, but you can use MultiTaskPredictor as a starting point."
    )

    _WRONG_FIELD_ERROR = (
        "MultiTaskPredictor expects instances that have a MetadataField "
        + "with the name 'task', containing the name of the task the instance is for."
    )

    def __init__(self, model: MultiTaskModel, dataset_reader: MultiTaskDatasetReader) -> None:
        if not isinstance(dataset_reader, MultiTaskDatasetReader):
            raise ConfigurationError(self._WRONG_READER_ERROR)

        if not isinstance(model, MultiTaskModel):
            raise ConfigurationError(
                "MultiTaskPredictor is designed to work only with MultiTaskModel."
            )

        super().__init__(model, dataset_reader)

        self.predictors = {}
        for name, head in model._heads.items():
            predictor_name = head.default_predictor
            predictor_class: Type[Predictor] = (
                Predictor.by_name(predictor_name) if predictor_name is not None else Predictor  # type: ignore
            )
            self.predictors[name] = predictor_class(model, dataset_reader.readers[name].inner)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        task_field = instance["task"]
        if not isinstance(task_field, MetadataField):
            raise ValueError(self._WRONG_FIELD_ERROR)
        task: str = task_field.metadata
        if not isinstance(self._dataset_reader, MultiTaskDatasetReader):
            raise ConfigurationError(self._WRONG_READER_ERROR)
        self._dataset_reader.readers[task].apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        task = json_dict["task"]
        del json_dict["task"]
        predictor = self.predictors[task]
        instance = predictor._json_to_instance(json_dict)
        instance.add_field("task", MetadataField(task))
        return instance

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        task_to_instances: Dict[str, List[Instance]] = collections.defaultdict(lambda: [])
        for instance in instances:
            task_field = instance["task"]
            if not isinstance(task_field, MetadataField):
                raise ValueError(self._WRONG_FIELD_ERROR)
            task: str = task_field.metadata
            if not isinstance(self._dataset_reader, MultiTaskDatasetReader):
                raise ConfigurationError(self._WRONG_READER_ERROR)
            self._dataset_reader.readers[task].apply_token_indexers(instance)
            task_to_instances[task].append(instance)

        outputs = []
        for task, instances in task_to_instances.items():
            outputs.extend(super().predict_batch_instance(instances))

        return outputs
