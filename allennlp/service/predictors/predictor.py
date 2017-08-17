import json
import os
from typing import Dict

from allennlp.common import Params, Registrable
from allennlp.common.params import replace_none
from allennlp.common.util import JsonDict
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Vocabulary
from allennlp.models import Model


class Predictor(Registrable):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """
    def __init__(self, model: Model, vocab: Vocabulary,
                 tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer]) -> None:
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: Params) -> 'Predictor':
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


# TODO(joelgrus): delete this function
def default_params() -> Dict[str, Params]:
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
        config['model']['text_field_embedder']['tokens']['pretrained_file'] = \
            'tests/fixtures/glove.6B.300d.sample.txt.gz'
        decomposable_attention_config = Params(replace_none(config))

    return {
            'machine-comprehension': bidaf_config,
            'semantic-role-labelling': srl_config,
            'textual-entailment': decomposable_attention_config
    }




def load_predictors(configs: Dict[str, Params] = None) -> Dict[str, Predictor]:
    # TODO(joelgrus): remove this
    if configs is None:
        configs = default_params()

    return {
            name: Predictor.by_name(name).from_config(config)
            for name, config in configs.items()
    }
