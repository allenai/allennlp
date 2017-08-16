import json
import os
from typing import Dict, Any, Optional, List

from allennlp.common import Params, Registrable, constants
from allennlp.common.params import replace_none
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Vocabulary
from allennlp.models import Model

JsonDict = Dict[str, Any]  # pylint: disable=invalid-name


class Servable(Registrable):
    def __init__(self, model: Model, vocab: Vocabulary, dataset_reader: DatasetReader):
        print("initializing", self)
        self.model = model
        self.vocab = vocab
        self.dataset_reader = dataset_reader

        try:
            self.tokenizer = dataset_reader._tokenizer  # pylint: disable=protected-access
        except AttributeError:
            self.tokenizer = WordTokenizer()

        try:
            self.token_indexers = dataset_reader._token_indexers  # pylint: disable=protected-access
        except AttributeError:
            self.token_indexers = {}

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: Params) -> 'Servable':
        dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        serialization_prefix = config['trainer']['serialization_prefix']
        vocab_dir = os.path.join(serialization_prefix, 'vocabulary')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = config["model"]
        model = Model.from_params(vocab, model_params)

        # TODO(joelgrus): use a GPU if appropriate
        # cuda_device = -1

        weights_file = os.path.join(serialization_prefix, "best.th")

        # TODO(joelgrus): get rid of this check once we have weights files for all the models
        #if os.path.exists(weights_file):
        #    model_state = torch.load(weights_file, map_location=device_mapping(cuda_device))
        #    model.load_state_dict(model_state)
        model.eval()

        return cls(model, vocab, dataset_reader)


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
