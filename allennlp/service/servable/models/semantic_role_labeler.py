from allennlp.data import Vocabulary, DatasetReader
from allennlp.data.fields import TextField, IndexField
from allennlp.models import Model
from allennlp.service.servable import Servable, JsonDict

import spacy

@Servable.register("srl")
class SemanticRoleLabelerServable(Servable):
    def __init__(self, model: Model, vocab: Vocabulary, dataset_reader: DatasetReader):
        super().__init__(model, vocab, dataset_reader)

        self.nlp = spacy.load('en')

    def predict_json(self, inputs: JsonDict) -> JsonDict:
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
