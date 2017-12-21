from typing import List, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register("semantic-role-labeling")
class SemanticRoleLabelerPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

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

    def _sentence_to_srl_instances(self, json_dict: JsonDict) -> Tuple[List[Instance], JsonDict]:

        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        words = [token.text for token in tokens]
        result_dict: JsonDict = {"words": words, "verbs": []}
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            if word.pos_ == "VERB":
                verb = word.text
                result_dict["verbs"].append(verb)
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(tokens, verb_labels)
                instances.append(instance)
        return instances, result_dict

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:

        batch_size = len(inputs)
        instances_per_sentence, return_dicts = zip(*[self._sentence_to_srl_instances(json)
                                                     for json in inputs])

        flattened_instances = sum(instances_per_sentence, [])
        batched_instances = group_by_count(flattened_instances, batch_size, None)
        batched_instances[-1] = [instance for instance in batched_instances
                                 if instance is not None]
        outputs = []
        for batch in batched_instances:
            outputs.extend(self._model.forward_on_instances(batch, cuda_device))

        sentence_index = 0
        for results in return_dicts:
            verbs_for_sentence: List[str] = results["verbs"]
            results["verbs"] = []
            for verb in verbs_for_sentence:

                output = outputs[sentence_index]
                tags = output['tags']
                description = SemanticRoleLabelerPredictor.make_srl_string(results["words"], tags)
                results["verbs"].append({
                        "verb": verb,
                        "description": description,
                        "tags": tags,
                })
                sentence_index += 1

            results["tokens"] = results["words"]

        return return_dicts

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
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
        instances, results = self._sentence_to_srl_instances(inputs)
        verbs_for_instances: List[str] = results["verbs"]
        results["verbs"] = []

        outputs = self._model.forward_on_instances(instances, cuda_device)

        for output, verb in zip(outputs, verbs_for_instances):
                tags = output['tags']
                description = SemanticRoleLabelerPredictor.make_srl_string(results["words"], tags)
                results["verbs"].append({
                        "verb": verb,
                        "description": description,
                        "tags": tags,
                })

        results["tokens"] = results["words"]
        return sanitize(results)
