import logging
from typing import Optional

from overrides import overrides

from allennlp.common import Params
from allennlp.experiments.driver import Driver
from allennlp.training import Model, Trainer
from allennlp.data import Dataset
from allennlp.data.dataset_readers import DatasetReader

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


class TrainDriver(Driver):
    def __init__(self,
                 model: Model,
                 train_data: Dataset,
                 validation_data: Optional[Dataset],
                 trainer: Trainer) -> None:
        self._model = model
        self._train_data = train_data
        self._validation_data = validation_data
        self._trainer = trainer

    @overrides
    def run(self):
        self._trainer.train_model(self._model, self._train_data, self._validation_data)
        self._model.save()

    @classmethod
    def from_params(cls, params: Params) -> 'TrainDriver':
        model = Model.from_params(params['model'])
        dataset_reader = DatasetReader.from_params(params['dataset_reader'])
        train_data_path = params['train_data_path']
        logger.info("Reading training data from %s", train_data_path)
        train_data = dataset_reader.read(train_data_path)
        validation_data_path = params['validation_data_path']
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        trainer = Trainer.from_params(params['training'])
        params.assert_empty(cls.__name__)
        return TrainDriver(model, train_data, validation_data, trainer)
