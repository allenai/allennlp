

import torch

from ..data.dataset import Dataset
from ..common.params import Params

class Trainer:

    """
    The Trainer class manages everything to do with training, loading and saving a model.
    """



    def __init__(self,
                 model: torch.nn.Module,
                 dataset: Dataset):


        self.dataset = dataset
        self.model = model


    def train_model(self):



    def save_model(self, file_path):

        state_dictionary = self.model.state_dict()
        torch.save(state_dictionary, file_path)



    @classmethod
    def from_params(cls, params: Params):

        dataset = params.pop("dataset")
        model = params.pop("model")
        return cls(dataset=dataset,
                   model=model)