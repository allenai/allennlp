import json
import os
from collections.abc import Iterable
from typing import Dict, Any, Optional, List

from allennlp.common import Params, Registrable, constants
from allennlp.common.params import replace_none
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Vocabulary
from allennlp.models import Model

import numpy as np
import torch

JsonDict = Dict[str, Any]  # pylint: disable=invalid-name


def sanitize(x: Any) -> Any:
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, np.number):
        return x.item()
    elif isinstance(x, Iterable):
        return [sanitize(x_i) for x_i in x]
    elif isinstance(x, dict):
        return {key: sanitize(value) for key, value in x.items()}
    else:
        return x


class Servable(Registrable):
    def __init__(self, model: Model, vocab: Vocabulary,
                 tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer]) -> None:
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: Params) -> 'Servable':
        dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        tokenizer = dataset_reader._tokenizer or WordTokenizer()  # pylint: disable=protected-access
        token_indexers = dataset_reader._token_indexers           # pylint: disable=protected-access

        serialization_prefix = config['trainer']['serialization_prefix']
        vocab_dir = os.path.join(serialization_prefix, 'vocabulary')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = config["model"]
        model = Model.from_params(vocab, model_params)

        # TODO(joelgrus): load weights from files
        model.eval()

        return cls(model, vocab, tokenizer, token_indexers)


class ServableCollection:
    """
    This represents the collection of models that are available to our command line tool or REST API.
    """
    def __init__(self, collection: Dict[str, Servable] = None) -> None:
        self.collection = collection if collection is not None else {}

    def get(self, key: str) -> Optional[Servable]:
        return self.collection.get(key)

    def register(self, key: str, servable: Servable):
        self.collection[key] = servable

    def list_available(self) -> List[str]:
        return list(self.collection.keys())


    # TODO: get rid of this
    @staticmethod
    def default() -> 'ServableCollection':
        with open('experiment_config/bidaf.json') as config_file:
            config = json.loads(config_file.read())
            config['trainer']['serialization_prefix'] = 'tests/fixtures/bidaf/serialization'
            config['model']['text_field_embedder']['tokens']['pretrained_file'] = \
                'tests/fixtures/glove.6B.100d.sample.txt.gz'
            bidaf_config = Params(replace_none(config))

        with open('experiment_config/semantic_role_labeler.json') as config_file:
            config = json.loads(config_file.read())
            config['trainer']['serialization_prefix'] = 'tests/fixtures/srl'
            config['model']['text_field_embedder']['tokens']['pretrained_file'] = \
                'tests/fixtures/glove.6B.100d.sample.txt.gz'
            srl_config = Params(replace_none(config))

        with open('experiment_config/decomposable_attention.json') as config_file:
            config = json.loads(config_file.read())
            config['trainer']['serialization_prefix'] = 'tests/fixtures/decomposable_attention'
            # TODO(joelgrus) once the correct config exists, just modify it
            constants.GLOVE_PATH = 'tests/fixtures/glove.6B.300d.sample.txt.gz'
            decomposable_attention_config = Params(replace_none(config))

        from allennlp.service.servable.models.bidaf import BidafServable
        from allennlp.service.servable.models.decomposable_attention import DecomposableAttentionServable
        from allennlp.service.servable.models.semantic_role_labeler import SemanticRoleLabelerServable

        all_models = {
                'bidaf': BidafServable.from_config(bidaf_config),
                'srl': SemanticRoleLabelerServable.from_config(srl_config),
                'snli': DecomposableAttentionServable.from_config(decomposable_attention_config),
        }  # type: Dict[str, Servable]

        return ServableCollection(all_models)

    @staticmethod
    def from_params(params: Params) -> 'ServableCollection':  # pylint: disable=unused-argument
        # TODO(joelgrus) implement this
        return ServableCollection.default()
