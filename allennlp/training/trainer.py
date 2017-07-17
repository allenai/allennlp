import os
import shutil
import torch

from allennlp.data.dataset import Dataset
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.tensor import data_structure_as_variables
from allennlp.training.model import Model
from allennlp.data.iterators import DataIterator


class Trainer:

    """
    The Trainer class manages everything to do with training a model.
    """

    def __init__(self,
                 iterator: DataIterator,
                 patience: int = 2,
                 batch_size: int = 32,
                 num_epochs: int = 20,
                 serialization_prefix: str = None,
                 save_models: bool = False,
                 cuda_device: int = -1) -> None:
        """
        Parameters
        ----------
        patience : int, optional (default=2)
            Number of epochs to be patient before early stopping.  I.e., if the ``validation_metric``
            does not improve for this many epochs, we will stop training.
        batch_size : int, optional (default=32)
            Batch size to use when training.
        num_epochs : int, optional (default=20)
            Number of training epochs.
        serialization_prefix : str, optional (default=None)
            Path to directory for saving and loading model files.  Must be set if ``save_models`` is ``True``.
        save_models: bool, optional (default=False)
            Whether or not to save the model during training.
        """
        self.iterator = iterator
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.serialization_prefix = serialization_prefix
        self.save_models = save_models
        self.cuda_device = cuda_device

        if self.save_models and not self.serialization_prefix:
            raise ConfigurationError("save_models = True but no 'serialization_prefix' provided.")

    def train_model(self, model: Model,
                    dataset: Dataset,
                    optimizer: torch.optim.Optimizer):

        for batch in self.iterator(dataset):
            # Convert all the input tensors into torch Tensors.
            # It correctly converts numpy types, eg. np.int32 -> torch.IntTensor.
            torch_batch = data_structure_as_variables(batch, cuda_device=self.cuda_device)
            model.state_dict()
            optimizer.zero_grad()
            output_dict = model.forward(torch_batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()

    def save_checkpoint(self,
                        model: Model,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        is_best: bool):
        model_path = os.path.join(self.serialization_prefix, "model_state")
        model_state = model.state_dict()
        torch.save(model_state, os.path.join(model_path, "epoch_{}.tar".format(epoch)))

        training_state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
        }
        torch.save(training_state, os.path.join(self.serialization_prefix,
                                                "training_state_epoch_{}.tar".format(epoch)))

        if is_best:
            shutil.copy(model_path, os.path.join(model_path, "best.tar"))



    @classmethod
    def from_params(cls, params: Params):
        iterator = DataIterator.from_params(params.pop("iterator", {}))
        kwargs = params.as_dict()
        return cls(iterator, **kwargs)
