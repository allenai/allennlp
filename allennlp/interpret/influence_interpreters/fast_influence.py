import json

from typing import List, Optional, Tuple, Union
import numpy as np
from torch import Tensor
from tqdm import tqdm
import torch
import torch.autograd as autograd
from torch.nn import Parameter

from allennlp.interpret.influence_interpreters.influence_interpreter import (
    InfluenceInterpreter,
)
from allennlp.predictors import Predictor
from allennlp.models.model import Model
from allennlp.data import DatasetReader, Batch, Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.data_loaders import DataLoader, MultiProcessDataLoader
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BertPooler

from allennlp.nn import util
from allennlp.interpret.influence_interpreters.influence_utils import faiss_utils
from allennlp.interpret.influence_interpreters.influence_utils import FAISSWrapper


@InfluenceInterpreter.register("fast-influence")
class FastInfluence(InfluenceInterpreter):
    """
    Registered as a `FastInfluence` with name "fast-influence". This is a simple influence function
    calculator. We simply go through every examples in train set to calculate the influence score, and uses
    recommended LiSSA algorithm (essentially a first-order Talyor approxmation) to approximate the inverse
    of Hessian used for influence score calculation. At best, we uses a single GPU for running the calculation.

    # Parameter
    predictor: `Predictor`
        Required. This is a wrapper around the model to be tested. We only assume only `Model` is not None.
    train_filepath: `str`
        Required. This is the file path to the train data
    test_filepath: `str`
        Required. This is the file path to the test data
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
    damping: float = 3e-3,
        Optional. This is a hyperparameter for LiSSA algorithm.
        A damping termed added in case the approximated Hessian (during LiSSA algorithm) has
        negative eigenvalues. This is a hyperparameter.
    num_samples: int = 1,
        Optional. This is a hyperparameter for LiSSA algorithm that we
        determine how many rounds of recursion process we would like to run for approxmation.
    recur_depth: Optional[Union[float, int]] = 0.25,
        Optional. This is a hyperparameter for LiSSA algorithm that we
        determine the recursion depth we would like to go through.
    scale: float = 1e4,
        Optional. This is a hyperparameter for LiSSA algorithm to tune such that the Taylor expansion converges.
    """

    def __init__(
        self,
        predictor: Predictor,
        train_filepath: str,
        test_filepath: str,
        train_dataset_reader: DatasetReader,
        # seq2vec_encoder: Optional[Seq2VecEncoder] = BertPooler("bert-base-uncased"),
        faiss_dataset_reader: Optional[DatasetReader] = None,
        faiss_dataset_wrapper: Optional[FAISSWrapper] = None,
        # faiss_token_indexer: Optional[TokenIndexer] = None,
        # faiss_tokenizer: Optional[Tokenizer] = None,
        # faiss_vocab: Optional[Vocabulary] = None,
        # faiss_text_field_embedder: Optional[TokenEmbedder] = None,
        test_dataset_reader: Optional[DatasetReader] = None,
        params_to_freeze: Optional[List[str]] = None,
        k: int = 20,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        device: int = -1,
        damping: float = 3e-3,
        num_samples: int = 1,
        scale: float = 1e4,
    ) -> None:
        super().__init__(
            predictor=predictor,
            train_dataset_reader=train_dataset_reader,
            test_dataset_reader=test_dataset_reader,
            train_filepath=train_filepath,
            test_filepath=test_filepath,
            params_to_freeze=params_to_freeze,
            k=k,
            device=device,
        )
        # self.seq2vec_encoder = seq2vec_encoder
        # vectorize the training set, so t
        # TODO (@Leo): might give user the option to choose whether to use FAISS
        self.faiss_wrapper = faiss_dataset_wrapper
        # self.faiss_index = faiss_utils.FAISSIndex(seq2vec_encoder.get_output_dim(), faiss_description)
        # self.faiss_token_indexer = faiss_token_indexer
        # self.faiss_vocab = faiss_vocab
        # self.faiss_tokenizer = faiss_tokenizer
        # self.faiss_text_field_embedder = faiss_text_field_embedder
        self._faiss_train_loader = MultiProcessDataLoader(
            faiss_dataset_reader, train_filepath, batch_size=128, shuffle=False
        )
        self._faiss_train_loader.index_with(self.faiss_wrapper.vocab)
        self._create_faiss_index()
        self.damping = damping
        self.num_samples = num_samples
        self.scale = scale

        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        self.weight_decay_ignores = ["bias", "LayerNorm.weight"] if weight_decay_ignores is None \
            else weight_decay_ignores

    def _create_faiss_index(self):
        # TODO: re-write this in AllenNLP fashion
        for inputs in tqdm(self._faiss_train_loader):

            features = self.faiss_wrapper(inputs)
            features = features.cpu().detach().numpy()
            self.faiss_wrapper.add(features)



