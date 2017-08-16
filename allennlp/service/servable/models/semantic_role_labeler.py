from typing import Dict

from allennlp.data import Vocabulary, Tokenizer, TokenIndexer
from allennlp.data.fields import TextField, IndexField
from allennlp.models import Model
from allennlp.service.servable import Servable, JsonDict

import spacy

@Servable.register("srl")
class SemanticRoleLabelerServable(Servable):
    def __init__(self, model: Model, vocab: Vocabulary,
                 tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer]) -> None:
        super().__init__(model, vocab, tokenizer, token_indexers)

        self.nlp = spacy.load('en', parser=False, vectors=False, entity=False)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        sentence = inputs["sentence"]
        tokens = self.tokenizer.tokenize(sentence)
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

        return results
