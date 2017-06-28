from typing import Any, Dict

import torch

from ..data.dataset import Dataset
from ..common.params import Params
from .optimizers import get_optimizer_from_params
class Trainer:

    """
    The Trainer class manages everything to do with training, loading and saving a model.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_params: Dict[str, Any],
                 dataset: Dataset):


        self.model = model
        self.optimizer = get_optimizer_from_params(self.model.parameters(), optimizer_params)
        self.dataset = dataset

    def train_model(self):



    def save_model(self, file_path):

        state_dictionary = self.model.state_dict()
        torch.save(state_dictionary, file_path)



    @classmethod
    def from_params(cls, params: Params):
        dataset = params.pop("dataset")
        model = params.pop("model")
        optimizer_params = params.pop("optimizer", Params({'type:':'adam'})).as_dict()

        return cls(dataset=dataset,
                   optimizer_params=optimizer_params,
                   model=model)