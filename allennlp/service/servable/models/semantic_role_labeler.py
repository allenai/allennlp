import os
from typing import Dict

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SrlReader
from allennlp.data.fields import TextField, IndexField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.models import SemanticRoleLabeler
from allennlp.service.servable import Servable, JSONDict

import spacy

class SemanticRoleLabelerServable(Servable):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 model: SemanticRoleLabeler) -> None:
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.model = model
        self.nlp = spacy.load('en', parser=False, vectors=False, entity=False)

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        sentence = inputs["sentence"]
        tokens = self.tokenizer.tokenize(sentence)
        text = TextField(tokens, token_indexers=self.token_indexers)

        results = {"verbs": []}  # type: JSONDict
        spacy_doc = self.nlp(sentence)
        for i, word in enumerate(spacy_doc):
            if word.pos_ == "VERB":
                verb_indicator = IndexField(i, text)
                output = self.model.tag(text, verb_indicator)
                results["verbs"].append({
                        "index": i,
                        "verb": word.text,
                        "tags": output["tags"],
                        "class_probabilities": output["class_probabilities"].tolist()
                })

        return results

    @classmethod
    def from_config(cls, config: Params) -> 'SemanticRoleLabelerServable':
        dataset_reader_params = config.pop("dataset_reader")
        assert dataset_reader_params.pop('type') == 'srl'
        dataset_reader = SrlReader.from_params(dataset_reader_params)

        serialization_prefix = config.pop('serialization_prefix')
        vocab_dir = os.path.join(serialization_prefix, 'vocabulary')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = config.pop("model")
        assert model_params.pop("type") == "srl"
        model = SemanticRoleLabeler.from_params(vocab, model_params)

        # TODO(joelgrus): load weights

        # pylint: disable=protected-access
        # use default WordTokenizer, since there's none in the experiment spec
        return SemanticRoleLabelerServable(tokenizer=WordTokenizer(),
                                           token_indexers=dataset_reader._token_indexers,
                                           model=model)
        # pylint: enable=protected-access
