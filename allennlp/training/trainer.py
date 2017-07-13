from allennlp.common import Params
from allennlp.data import Dataset
from allennlp.training.model import Model


class Trainer:
    # Mark is working on actually implementing this - this is just a suggested API to make the
    # driver simple.
    def train_model(self,
                    model: Model,
                    train_dataset: Dataset,
                    validation_dataset: Dataset):
        pass
    @classmethod
    def from_params(cls, params: Params) -> 'Trainer':
        pass
