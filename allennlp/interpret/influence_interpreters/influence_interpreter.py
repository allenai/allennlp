from typing import List

import numpy
import torch

from allennlp.common import Registrable
from allennlp.common.util import JsonDict
from allennlp.nn import util
from allennlp.models.model import Model
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader, DataLoader, PyTorchDataLoader # , MultiProcessDataLoader


class InfluenceInterpreter(Registrable):
    """
    A `SaliencyInterpreter` interprets an AllenNLP Predictor's outputs by assigning an influence
    score to each training instance with respect to each input.
    """

    def __init__(self,
                 predictor: Predictor,
                 train_dataset_reader: DatasetReader,
                 test_dataset_reader: DatasetReader,
                 train_filepath: str,
                 test_filepath: str,
                 train_batch_size: int,
                 k: int = 20,
                 device: int = -1) -> None:
        self.predictor = predictor
        self.model = self.predictor._model
        self.vocab = self.model.vocab
        self.train_dataset_reader = train_dataset_reader
        self.test_dataset_reader = test_dataset_reader or train_dataset_reader
        self._train_filepath = train_filepath
        self._test_filepath = test_filepath

        self._train_set = self.train_dataset_reader.read(train_filepath)
        self._train_set.index_instances(self.vocab)
        self._test_set = self.test_dataset_reader.read(test_filepath)
        self._test_set.index_instances(self.vocab)
        self._train_batch_size = train_batch_size
        self._train_loader = PyTorchDataLoader(self._train_set, batch_size=self._train_batch_size)
        self._k = k
        self._device = torch.device(f"cuda:{int(device)}" if torch.cuda.is_available() and not device >= 0 else "cpu")
        self.model.to(self._device)

        # so far, we assume all parameters are tuned during training
        # we use freeze, because when model is loaded from archive, the requires_grad
        # flag will be re-initialized to be true
        self._freeze_model()
        # self._name2params = self._get_tunable_params()
        # self._used_name2params = None  # some parameters might not be used.
        # this is not set until we actually run the calculation, because some parameters might not be used.
        self._used_params = None
        self._used_params_name = None

    def influence_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        This function finds influence scores for each training examples with respect to each input.

        # Parameters

        inputs : `JsonDict`
            The input you want to interpret (the same as the argument to a Predictor, e.g., predict_json()).

        # Returns

        interpretation : `JsonDict`
            Contains a sorted list (length = k) of most influenctial training examples for each
            instance in the inputs JsonDict, e.g., `{instance_1: ..., instance_2, ...}`.
            Each of those entries has a sorted list of length k in the format of
            `[(train_instance_1, score_1), ..., (train_instance_k, score_k),]`
        """
        raise NotImplementedError("Implement this for saliency interpretations")

    def _freeze_model(self):
        """
        This method intends to freeze parts (or all) of the model.
        :return:
        """
        # TODO: finish it
        pass

    def _get_tunable_params(self):
        """
        This function return a list of `torch.nn.parameter.Parameter`
        """
        return dict([(n, p) for n, p in self.predictor._model.named_parameters() if p.requires_grad])

