from typing import Optional
import logging
import os
import shutil

import torch
import tqdm
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
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
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Dataset,
                 validation_dataset: Optional[Dataset] = None,
                 patience: int = 2,
                 batch_size: int = 32,
                 num_epochs: int = 20,
                 serialization_prefix: Optional[str] = None,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimzier``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on.
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
        cuda_device : int, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        """
        self._model = model
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset

        self._iterator = iterator
        self._optimizer = optimizer
        self._patience = patience
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._serialization_prefix = serialization_prefix
        self._cuda_device = cuda_device

        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

    @overrides
    def run(self):
        epoch_counter = 0
        # Resume from serialization path if it contains a saved model.
        if self._serialization_prefix is not None:
            if any(["model_state_epoch_" in x
                    for x in os.listdir(self._serialization_prefix)]):
                epoch_counter = self._restore_checkpoint()

        for epoch in range(epoch_counter, self._num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_generator = self._iterator(self._train_dataset, num_epochs=1)

            for batch in tqdm.tqdm(train_generator):
                tensor_batch = arrays_to_variables(batch, self._cuda_device)
                self._optimizer.zero_grad()
                output_dict = self._model.forward(tensor_batch)
                try:
                    loss = output_dict["loss"]
                    loss.backward()
                    train_loss += loss.data.numpy()
                except KeyError:
                    raise ConfigurationError("The model you are trying to optimize does not contain a"
                                             " 'loss' key in the output of model.forward(inputs).")
                self._optimizer.step()

            val_generator = self._iterator(self._validation_dataset, num_epochs=1)

            for batch in tqdm.tqdm(val_generator):
                tensor_batch = arrays_to_variables(batch, self._cuda_device)
                val_output_dict = self._model.forward(tensor_batch)
                loss = val_output_dict["loss"]
                val_loss += loss.data.numpy()
                # TODO(): metrics here.

            logger.log("Training Loss: %3f    Validation Loss: %3f ", train_loss, val_loss)
            self._save_checkpoint(self._model, self._optimizer, epoch)

    def _save_checkpoint(self,
                         model: Model,
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         is_best: Optional[bool] = None):
        model_path = os.path.join(self.serialization_prefix, "model_state_epoch_{}.th".format(epoch))
        model_state = model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch, 'optimizer': optimizer.state_dict()}
        torch.save(training_state, os.path.join(self.serialization_prefix,
                                                "training_state_epoch_{}.th".format(epoch)))
        if is_best:
            shutil.copy(model_path, os.path.join(model_path, "best.th"))

    def _restore_checkpoint(self):

        serialization_files = os.listdir(self._serialization_prefix)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) for x in model_checkpoints])

        model_state = torch.load(os.path.join(self.serialization_prefix,
                                              "model_state_epoch_{}.th".format(epoch_to_load)))
        training_state = torch.load(os.path.join(self.serialization_prefix,
                                                 "training_state_epoch_{}.th".format(epoch_to_load)))
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(training_state["optimizer"])
        return training_state["epoch"]

    @classmethod
    def from_params(cls, params: Params) -> 'TrainDriver':

        model = Model.from_params(params.pop('model'))
        dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))

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
