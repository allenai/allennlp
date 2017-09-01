from typing import Dict

from allennlp.common import Registrable
from allennlp.common.util import JsonDict
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.models.archival import Archive

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'srl': 'semantic-role-labeling',
        'decomposable_attention': 'textual-entailment',
        'bidaf': 'machine-comprehension',
        'simple_tagger': 'simple-tagger'
}

class Predictor(Registrable):
    """
    a ``Predictor`` is a thin wrapper around an AllenNLP model that handles JSON -> JSON predictions
    that can be used for serving models through the web API or making predictions in bulk.
    """
    def __init__(self,
                 model: Model,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer]) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        raise NotImplementedError()

    @classmethod
    def from_archive(cls, archive: Archive, predictor_name: str = None) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, the default one for the model will be used.
        """
        config = archive.config

        dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        tokenizer = dataset_reader._tokenizer or WordTokenizer()  # pylint: disable=protected-access
        token_indexers = dataset_reader._token_indexers           # pylint: disable=protected-access

        model = archive.model
        model.eval()

        model_name = config.get("model").get("type")
        predictor_name = predictor_name or DEFAULT_PREDICTORS[model_name]
        return Predictor.by_name(predictor_name)(model, tokenizer, token_indexers)
