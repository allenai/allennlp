import logging
from typing import Optional

from overrides import overrides

from allennlp.common import Params
from allennlp.experiments.driver import Driver
from allennlp.data.data_iterator import DataIterator
from allennlp.training import Model
from allennlp.data import Dataset
from allennlp.data.dataset_reader import DatasetReader
from allennlp.experiments.registry import Registry

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


@Registry.register_driver("train")
class TrainDriver(Driver):
    def __init__(self,
                 model: Model,
                 train_data: Dataset,
                 validation_data: Optional[Dataset],
                 iterator: DataIterator,
                 patience: int = 2,
                 batch_size: int = 32,
                 num_epochs: int = 20,
                 serialization_prefix: Optional[str] = None,
                 cuda_device: int = -1) -> None:

        self._model = model
        self._train_data = train_data
        self._validation_data = validation_data

        self._iterator = iterator
        self._patience = patience
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._serialization_prefix = serialization_prefix
        self._cuda_device = cuda_device

    @overrides
    def run(self):
        self._model.save()

    @classmethod
    def from_params(cls, params: Params) -> 'TrainDriver':

        model = Model.from_params(params['model'])
        dataset_reader = DatasetReader.from_params(params['dataset_reader'])

        train_data_path = params.pop('train_data_path')
        logger.info("Reading training data from %s", train_data_path)
        train_data = dataset_reader.read(train_data_path)

        validation_data_path = params.pop('validation_data_path')
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)

        iterator = DataIterator.from_params(params.pop("iterator"))

        params.assert_empty(cls.__name__)
        return TrainDriver(model, train_data, validation_data, iterator)
