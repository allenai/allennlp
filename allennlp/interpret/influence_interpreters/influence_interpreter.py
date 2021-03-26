from typing import List, Optional

import torch

from allennlp.common import Registrable
from allennlp.models.model import Model
from allennlp.predictors import Predictor
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.data_loaders import MultiProcessDataLoader


class InfluenceInterpreter(Registrable):
    """
    A `SaliencyInterpreter` interprets an AllenNLP Predictor's outputs by assigning an influence
    score to each training instance with respect to each test input.

    # Parameter
    predictor: `Predictor`
        Required. This is a wrapper around the model to be tested. We only assume only `Model` is not None.
    train_filepath: `str`
        Required. This is the file path to the train data
    train_dataset_reader: `DatasetReader`
        Required. This is the dataset reader to read the train set file
    test_dataset_reader: `Optional[DatasetReader]` = None,
        Optional. This is the dataset reader to read the test set file. If not provided, we would uses the
        `train_dataset_reader`
    params_to_freeze: Optional[List[str]] = None
        Optional. This is a provided list of string that for freezeing the parameters. Expectedly, each string
        is a substring within the paramter name you intend to freeze.
    k: int = 20
        Optional. To demonstrate each test data, we found it most informative to just provide `k` examples with the
        highest and lowest influence score. If not provided, we set to 20.
    device: int = -1,
        Optional. The index of GPU device we want to calculate scores on. If not provided, we uses -1
        which correspond to using CPU.
    """

    def __init__(
        self,
        predictor: Predictor,
        train_data_path: str,
        train_dataset_reader: DatasetReader,
        test_dataset_reader: Optional[DatasetReader] = None,
        params_to_freeze: List[str] = None,
        k: int = 20,
        device: int = -1,
    ) -> None:

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
            self.train_dataset_reader, train_data_path, batch_size=1
        )
        self._train_loader.set_target_device(self._device)
        self._train_loader.index_with(self.vocab)
        self.train_instances = [instance for instance in self._train_loader._instances]

        # Number of supporting training instances has to be less than the size of train set
        self._k = min(k, len(self._train_loader))

        self.model.to(self._device)

        # so far, we assume all parameters are tuned during training
        # we use freeze, because when model is loaded from archive, the requires_grad
        # flag will be re-initialized to be true
        if params_to_freeze is not None:
            self.freeze_model(self.model, params_to_freeze, verbose=True)
        # this is not set until we actually run the calculation, because some parameters might not be used.
        self._used_params: List = []
        self._used_params_name: List = []

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
