from typing import List, Optional

import torch

from allennlp.common import Registrable
from allennlp.common.util import JsonDict
from allennlp.models.model import Model
from allennlp.predictors import Predictor
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.data_loaders import MultiProcessDataLoader


class InfluenceInterpreter(Registrable):
    """
    A `SaliencyInterpreter` interprets an AllenNLP Predictor's outputs by assigning an influence
    score to each training instance with respect to each test input.
    """

    def __init__(
        self,
        predictor: Predictor,
        train_filepath: str,
        test_filepath: str,
        train_dataset_reader: DatasetReader,
        test_dataset_reader: Optional[DatasetReader] = None,
        params_to_freeze: List[str] = None,
        k: int = 20,
        device: int = -1,
    ) -> None:
        if type(device) is not int:
            raise ValueError("'device' has to be int, -1 for cpu")
        if type(device) is not int:
            raise ValueError("'k' (i.e. number of supporting examples) has to be int")

        self.predictor = predictor
        self.model: Model = self.predictor._model
        self.vocab = self.model.vocab
        self.train_dataset_reader = train_dataset_reader
        self.test_dataset_reader = test_dataset_reader or train_dataset_reader

        self._device = torch.device(
            f"cuda:{int(device)}" if torch.cuda.is_available() and not device >= 0 else "cpu"
        )
        # Dataloaders for going through train/test set (1 by 1)
        self._train_loader = MultiProcessDataLoader(
            self.train_dataset_reader, train_filepath, batch_size=1
        )
        self._train_loader.set_target_device(self._device)
        self._train_loader.index_with(self.vocab)
        self._test_loader = MultiProcessDataLoader(
            self.test_dataset_reader, test_filepath, batch_size=1
        )
        self._test_loader.set_target_device(self._device)
        self._test_loader.index_with(self.vocab)

        # Number of supporting training instances has to be lass than the size of train set
        self._k = min(k, len(self._train_loader))

        self.model.to(self._device)

        # so far, we assume all parameters are tuned during training
        # we use freeze, because when model is loaded from archive, the requires_grad
        # flag will be re-initialized to be true
        if params_to_freeze is not None:
            self.freeze_model(self.model, params_to_freeze, verbose=True)
        # self._used_name2params = None  # some parameters might not be used.
        # this is not set until we actually run the calculation, because some parameters might not be used.
        self._used_params = None
        self._used_params_name = None

    def influence_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        This function finds influence scores for each training examples with respect to each input.

        # Parameters

        inputs  :    `JsonDict`
            The input you want to interpret (the same as the argument to a Predictor, e.g., predict_json()).

        # Returns

        interpretation  :   `JsonDict`
            Contains a sorted list (length = k) of most influenctial training examples for each
            instance in the inputs JsonDict, e.g., `{instance_1: ..., instance_2, ...}`.
            Each of those entries has a sorted list of length k in the format of
            `[(train_instance_1, score_1), ..., (train_instance_k, score_k),]`
        """
        raise NotImplementedError("Implement this for saliency interpretations")

    @staticmethod
    def freeze_model(model, params_to_freeze: List[str], verbose: bool = True):
        """
        This method intends to freeze parts (or all) of the model.

        params_to_freeze:
            list of substrings of model's parameter names (i.e. string)
            TODO: instead use regular expression?
        verbose:
            whether we print the trainable parameters
        """
        for n, p in model.named_parameters():
            if any(pfreeze in n for pfreeze in params_to_freeze):
                p.requires_grad = False
        if verbose:
            num_trainable_params = sum(
                [p.numel() for n, p in model.named_parameters() if p.requires_grad]
            )
            trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
            print(
                f"Params Trainable: {num_trainable_params}\n\t" + "\n\t".join(trainable_param_names)
            )
