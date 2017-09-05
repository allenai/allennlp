from typing import List

from overrides import overrides
import spacy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register("semantic-role-labeling")
class SemanticRoleLabelerPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.nlp = spacy.load('en', parser=False, vectors=False, entity=False)

    @staticmethod
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
                    chunk.append(tag[2:] + ": " + token)
                elif tag == "O":
                    frame.append(token)

        if chunk:
            frame.append("[" + " ".join(chunk) + "]")

        return " ".join(frame)

    def _json_to_instance(self, json: JsonDict) -> Instance:
        # We're overriding `predict_json` directly, so we don't need this.  But I'd rather have a
        # useless stub here then make the base class throw a RuntimeError instead of a
        # NotImplementedError - the checking on the base class is worth it.
        raise RuntimeError("this should never be called")

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like

        .. code-block:: js

            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]}
        """
        sentence = inputs["sentence"]

        spacy_doc = self.nlp(sentence)
        words = [token.text for token in spacy_doc]
        results: JsonDict = {"words": words, "verbs": []}
        for i, word in enumerate(spacy_doc):
            if word.pos_ == "VERB":
                verb = word.text
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(words, verb_labels)
                output = self._model.decode(self._model.forward_on_instance(instance))
                tags = output['tags']

                description = SemanticRoleLabelerPredictor.make_srl_string(words, tags)

                results["verbs"].append({
                        "verb": verb,
                        "description": description,
                        "tags": tags,
                })

        results["tokens"] = [word.text for word in spacy_doc]

        return sanitize(results)
