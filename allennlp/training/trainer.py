import logging
import os
import shutil
from typing import Optional

import torch
from torch.nn.utils.clip_grad import clip_grad_norm
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.tensor import arrays_to_variables
from allennlp.data import Dataset
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Trainer:
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Dataset,
                 validation_dataset: Optional[Dataset] = None,
                 patience: int = 2,
                 num_epochs: int = 20,
                 serialization_prefix: Optional[str] = None,
                 cuda_device: int = -1,
                 grad_norm: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : int, optional (default=2)
            Number of epochs to be patient before early stopping.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_prefix : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device : int, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm: float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        """
        self._model = model
        self._iterator = iterator
        self._optimizer = optimizer
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset

        self._patience = patience  # TODO(Mark): add this to training loop with validation metrics.
        self._num_epochs = num_epochs
        self._serialization_prefix = serialization_prefix
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm

        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)

    def train(self) -> None:
        epoch_counter = 0
        # Resume from serialization path if it contains a saved model.
        if self._serialization_prefix is not None:
            if any(["model_state_epoch_" in x
                    for x in os.listdir(self._serialization_prefix)]):
                epoch_counter = self._restore_checkpoint()

        for epoch in range(epoch_counter, self._num_epochs):
            train_loss = 0.0
            val_loss = 0.0

            # Set the model to "train" mode.
            self._model.train()
            train_generator = self._iterator(self._train_dataset, num_epochs=1)
            for batch in tqdm.tqdm(train_generator):
                tensor_batch = arrays_to_variables(batch, self._cuda_device)
                self._optimizer.zero_grad()
                output_dict = self._model.forward(**tensor_batch)
                try:
                    loss = output_dict["loss"]
                    loss.backward()
                    train_loss += loss.data.numpy()
                except KeyError:
                    raise ConfigurationError("The model you are trying to optimize does not contain a"
                                             " 'loss' key in the output of model.forward(inputs).")

                if self._grad_norm:
                    clip_grad_norm(self._model.parameters(), self._grad_norm)

                self._optimizer.step()

            if self._validation_dataset is not None:
                # Switch to evaluation mode.
                self._model.eval()
                val_generator = self._iterator(self._validation_dataset, num_epochs=1)
                for batch in tqdm.tqdm(val_generator):
                    tensor_batch = arrays_to_variables(batch, self._cuda_device)
                    val_output_dict = self._model.forward(tensor_batch)
                    loss = val_output_dict["loss"]
                    val_loss += loss.data.numpy()

            # TODO(Mark): Add user specified metrics here, maybe a "metrics" key?
            logger.info("Training Loss: %3f    Validation Loss: %3f ", train_loss, val_loss)
            if self._serialization_prefix:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self,
                         epoch: int,
                         is_best: Optional[bool] = None) -> None:
        """
        Parameters
        ----------
        epoch : int, required.
            The epoch of training.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        model_path = os.path.join(self._serialization_prefix, "model_state_epoch_{}.th".format(epoch))
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch, 'optimizer': self._optimizer.state_dict()}
        torch.save(training_state, os.path.join(self._serialization_prefix,
                                                "training_state_epoch_{}.th".format(epoch)))
        if is_best:
            logger.info("Best validation performance so far. "
                        "Copying weights to %s/best.th'.", self._serialization_prefix)
            shutil.copy(model_path, os.path.join(model_path, "best.th"))

    def _restore_checkpoint(self) -> int:
        """
        Restores a model from a serialization_prefix to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        Returns
        -------
        epoch: int
            The epoch at which to resume training.
        """
        if not self._serialization_prefix:
            raise ConfigurationError("serialization_prefix not specified - cannot "
                                     "restore a model without a directory path.")

        serialization_files = os.listdir(self._serialization_prefix)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) for x in model_checkpoints])

        model_state = torch.load(os.path.join(self._serialization_prefix,
                                              "model_state_epoch_{}.th".format(epoch_to_load)))
        training_state = torch.load(os.path.join(self._serialization_prefix,
                                                 "training_state_epoch_{}.th".format(epoch_to_load)))
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(training_state["optimizer"])
        return training_state["epoch"]

    @classmethod
    def from_params(cls,
                    model: Model,
                    optimizer: torch.optim.Optimizer,
                    iterator: DataIterator,
                    train_dataset: Dataset,
                    validation_dataset: Optional[Dataset],
                    params: Optional[Params] = None) -> 'Trainer':

        params = params or Params({})
        patience = params.pop("patience", 2)
        num_epochs = params.pop("num_epochs", 20)
        serialization_prefix = params.pop("serialization_prefix", None)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        params.assert_empty(cls.__name__)
        return Trainer(model, optimizer, iterator,
                       train_dataset, validation_dataset,
                       patience=patience,
                       num_epochs=num_epochs,
                       serialization_prefix=serialization_prefix,
                       cuda_device=cuda_device,
                       grad_norm=grad_norm)
