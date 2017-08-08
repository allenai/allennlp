from typing import Dict, Any  # pylint: disable=unused-import

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.data.fields import TextField, IndexField
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import SemanticRoleLabeler
from allennlp.service.servable import Servable, JSONDict

class SemanticRoleLabelerServable(Servable):
    def __init__(self):
        dataset = SrlReader().read('tests/fixtures/conll_2012/')
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset

        params = Params({
                "text_field_embedder": {
                        "tokens": {
                                "type": "embedding",
                                "embedding_dim": 5
                                }
                        },
                "stacked_encoder": {
                        "type": "lstm",
                        "input_size": 6,
                        "hidden_size": 7,
                        "num_layers": 2
                        }
                })

        self.model = SemanticRoleLabeler.from_params(self.vocab, params)

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        tokenizer = WordTokenizer()

        sentence = tokenizer.tokenize(inputs["sentence"])
        text = TextField(sentence, token_indexers={"tokens": SingleIdTokenIndexer()})
        results = {"idx": []}  # type: Dict[str, Any]
        for i in range(len(sentence)):
            verb_indicator = IndexField(i, text)
            output = self.model.tag(text, verb_indicator)
            output["class_probabilities"] = output["class_probabilities"].tolist()
            results["idx"].append(output)

        return results
