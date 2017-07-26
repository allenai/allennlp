from typing import Optional
import logging
import os
import shutil

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.experiments.driver import Driver
from allennlp.data.data_iterator import DataIterator
from allennlp.training import Model
from allennlp.training.optimizers import get_optimizer_from_params
from allennlp.data import Dataset
from allennlp.data.dataset_reader import DatasetReader
from allennlp.experiments.registry import Registry
from allennlp.common.tensor import arrays_to_variables

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Registry.register_driver("train")
class TrainDriver(Driver):
    def __init__(self,
                 model: Model,
                 train_data: Dataset,
                 validation_data: Optional[Dataset],
                 iterator: DataIterator,
                 optimizer: torch.optim.Optimizer,
                 patience: int = 2,
                 batch_size: int = 32,
                 num_epochs: int = 20,
                 serialization_prefix: Optional[str] = None,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        patience : int, optional (default=2)
            Number of epochs to be patient before early stopping.  I.e., if the ``validation_metric``
            does not improve for this many epochs, we will stop training.
        batch_size : int, optional (default = 32)
            Batch size to use when training.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_prefix : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        """

        self._model = model
        self._train_data = train_data
        self._validation_data = validation_data

        self._iterator = iterator
        self._optimizer = optimizer
        self._patience = patience
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._serialization_prefix = serialization_prefix
        self._cuda_device = cuda_device

    @overrides
    def run(self):

        train_generator = self._iterator(self._train_data, self._num_epochs)

        for batch in train_generator:
            tensor_batch = arrays_to_variables(batch, self._cuda_device)

        self._model.save()

    def save_checkpoint(self,
                        model: Model,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        is_best: bool):
        model_path = os.path.join(self.serialization_prefix, "model_state")
        model_state = model.state_dict()
        torch.save(model_state, os.path.join(model_path, "epoch_{}.th".format(epoch)))

        training_state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
        }
        torch.save(training_state, os.path.join(self.serialization_prefix,
                                                "training_state_epoch_{}.th".format(epoch)))

        if is_best:
            shutil.copy(model_path, os.path.join(model_path, "best.th"))

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

        optimizer = get_optimizer_from_params(model.parameters(), params)

        params.assert_empty(cls.__name__)
        return TrainDriver(model, train_data, validation_data, iterator, optimizer)
