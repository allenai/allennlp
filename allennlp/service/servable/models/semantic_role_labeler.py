from typing import Dict, Any  # pylint: disable=unused-import

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.data.fields import TextField, IndexField
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.models import SemanticRoleLabeler
from allennlp.service.servable import Servable, JSONDict

class SemanticRoleLabelerServable(Servable):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 model: SemanticRoleLabeler) -> None:
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.model = model

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        sentence = self.tokenizer.tokenize(inputs["sentence"])
        text = TextField(sentence, token_indexers=self.token_indexers)
        # TODO(joelgrus) use spacy to identify verbs
        results = {"idx": []}  # type: Dict[str, Any]
        for i in range(len(sentence)):
            verb_indicator = IndexField(i, text)
            output = self.model.tag(text, verb_indicator)
            output["class_probabilities"] = output["class_probabilities"].tolist()
            results["idx"].append(output)

        return results

    @classmethod
    def from_params(cls, params: Params) -> 'SemanticRoleLabelerServable':
        tokenizer = Tokenizer.from_params(params.pop("tokenizer"))

        token_indexers = {}
        token_indexer_params = params.pop('token_indexers')
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)

        vocab_dir = params.pop('vocab_dir')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = params.pop("model")
        assert model_params.pop("type") == "semantic_role_labeler"
        model = SemanticRoleLabeler.from_params(vocab, model_params)

        return SemanticRoleLabelerServable(tokenizer=tokenizer,
                                           token_indexers=token_indexers,
                                           model=model)
