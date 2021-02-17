from overrides import overrides
from typing import Type

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("multitask")
class MultiTaskPredictor(Predictor):
    """
    Predictor for multitask models.

    Registered as a `Predictor` with name "multitask".
    """

    _WRONG_READER_ERROR = (
        "MultitaskPredictor is designed to work with MultiTaskDatasetReader. "
        + "If you have a different DatasetReader, you have to write your own "
        + "Predictor, but you can use MultitaskPredictor as a starting point."
    )

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        from allennlp.common.checks import ConfigurationError
        from allennlp.models.multitask import MultiTaskModel
        from allennlp.data.dataset_readers.multitask import MultiTaskDatasetReader

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
            self.predictors[name] = predictor_class(model, dataset_reader.readers[name])

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        from allennlp.data.fields import MetadataField
        from allennlp.common.util import sanitize
        from allennlp.data.dataset_readers import MultiTaskDatasetReader
        from allennlp.common.checks import ConfigurationError

        task_field = instance.get("task")
        if not isinstance(task_field, MetadataField):
            raise ValueError(
                "MultiTaskPredictor expects instances that have a MetadataField "
                "with the name 'task', containing the name of the task the instance is for."
            )
        task: str = task_field.metadata
        if not isinstance(self._dataset_reader, MultiTaskDatasetReader):
            raise ConfigurationError(self._WRONG_READER_ERROR)
        self._dataset_reader.readers[task].apply_token_indexers(instance)
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        from allennlp.data.fields import MetadataField

        task = json_dict["task"]
        del json_dict["task"]
        predictor = self.predictors[task]
        instance = predictor._json_to_instance(json_dict)
        instance.add_field("task", MetadataField(task))
        return instance
