from typing import Dict

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Tokenizer, TokenIndexer
from allennlp.data.fields import TextField, IndexField
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor

import spacy

@Predictor.register("semantic-role-labeling")
class SemanticRoleLabelerPredictor(Predictor):
    def __init__(self, model: Model,
                 tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer]) -> None:
        super().__init__(model, tokenizer, token_indexers)

        self.nlp = spacy.load('en', parser=False, vectors=False, entity=False)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        sentence = inputs["sentence"]
        tokens = self.nlp.tokenizer(sentence)
        text = TextField(tokens, token_indexers=self.token_indexers)

        results = {"verbs": []}  # type: JsonDict
        spacy_doc = self.nlp(sentence)
        for i, word in enumerate(spacy_doc):
            if word.pos_ == "VERB":
                verb_indicator = IndexField(i, text)
                output = self.model.tag(text, verb_indicator)
                results["verbs"].append({
                        "index": i,
                        "verb": word.text,
                        "tags": output["tags"],
                        "class_probabilities": output["class_probabilities"]
                })

        return sanitize(results)
