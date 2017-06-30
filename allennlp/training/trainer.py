from typing import Any, Dict

import torch

from ..data.dataset import Dataset
from ..common.params import Params
from .optimizers import get_optimizer_from_params


class Trainer:

    """
    The Trainer class manages everything to do with training a model.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_params: Dict[str, Any],
                 dataset: Dataset,
                 patience: int,
                 batch_size: int = 32,
                 num_epochs: int = 20,
                 model_serialization_prefix: str = None,
                 cuda_device: int = -1):

        """
        Parameters
        ----------
        batch_size: int, optional (default=32)
            Batch size to use when training.
        num_epochs: int, optional (default=20)
            Number of training epochs.
        optimizer_params: str or Dict[str, Any], optional (default='adam')
            If this is a str, it must correspond to an optimizer available in Pytorch (see the list in
            :mod:`allennlp.training.optimizers`).  If it is a dictionary, it must contain a "type" key,
            with a value that is one of the optimizers in that list.  The remaining parameters in the
            dict are passed as kwargs to the optimizer's constructor.
        patience: int, optional (default=1)
            Number of epochs to be patient before early stopping.  I.e., if the ``validation_metric``
            does not improve for this many epochs, we will stop training.
        model_serialization_prefix: str, optional (default=None)
            Prefix for saving and loading model files.  Must be set if ``save_models`` is ``True``.
        """
        self.model = model
        self.optimizer = get_optimizer_from_params(self.model.parameters(), optimizer_params)
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model_serialization_prefix = model_serialization_prefix
        self.cuda_device = cuda_device

    def train_model(self):

        for batch in self.dataset.as_arrays():

            # Convert all the input tensors into torch Tensors.
            # from_numpy will correctly convert types, eg. np.int32 -> torch.IntTensor.
            torch_batch = {name: torch.from_numpy(array) for (name, array) in batch.items()}

            if self.cuda_device > -1:
                torch_batch = {name: array.cuda(self.cuda_device) for (name, array) in torch_batch.items()}

            self.optimizer.zero_grad()
            model_outputs, loss = self.model.forward(torch_batch)
            loss.backward()
            self.optimizer.step()

    def save_model(self, file_path):
        state_dictionary = self.model.state_dict()
        torch.save(state_dictionary, file_path)

    @classmethod
    def from_params(cls, params: Params):
        dataset = params.pop("dataset")
        model = params.pop("model")
        optimizer_params = params.pop("optimizer", Params({'type:': 'adam'})).as_dict()

        return cls(dataset=dataset,
                   optimizer_params=optimizer_params,
                   model=model)