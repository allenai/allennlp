from typing import Dict, List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Tokenizer, TokenIndexer
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor

import spacy

@Predictor.register("semantic-role-labeling")
class SemanticRoleLabelerPredictor(Predictor):
    def __init__(self, model: Model,
                 tokenizer: Tokenizer, token_indexers: Dict[str, TokenIndexer]) -> None:
        super().__init__(model, tokenizer, token_indexers)

        self.nlp = spacy.load('en', parser=False, vectors=False, entity=False)

    def make_srl_string(words: List[str], tags: List[str]) -> str:
        frame = []
        chunk = []

        for (token, tag) in zip(words, tags):
            if tag.startswith("I-"):
                chunk.append(token)
            else:
                if chunk:
                    frame.append("[" + " ".join(chunk) + "]")
                    chunk = []

                if tag.startswith("B-"):
                    chunk.append(tag + " " + token)
                elif tag == "O":
                    frame.append(token)

        if chunk:
            frame.append("[" + " ".join(chunk) + "]")

        return " ".join(frame)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        sentence = inputs["sentence"]
        tokens = self.nlp.tokenizer(sentence)
        text = TextField(tokens, token_indexers=self.token_indexers)

        results = {"verbs": []}  # type: JsonDict
        spacy_doc = self.nlp(sentence)
        words = [token.text for token in spacy_doc]
        for i, word in enumerate(spacy_doc):
            if word.pos_ == "VERB":
                verb_labels = [0 for _ in tokens]
                verb_labels[i] = 1
                verb_indicator = SequenceLabelField(verb_labels, text)
                output = self.model.tag(text, verb_indicator)

                verb = word.text
                tags = output["tags"]
                description = SemanticRoleLabelerPredictor.make_srl_string(words, tags)

                results["verbs"].append({
                        "verb": verb,
                        "description": description,
                        "tags": tags,
                })

        return sanitize(results)
