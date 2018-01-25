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

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        raise NotImplementedError("The SRL model uses a different API for creating instances.")

    def _sentence_to_srl_instances(self, json_dict: JsonDict) -> Tuple[List[Instance], JsonDict]:
        """
        The SRL model has a slightly different API from other models, as the model is run
        forward for every verb in the sentence. This means that for a single sentence, we need
        to generate a ``List[Instance]``, where the length of this list corresponds to the number
        of verbs in the sentence. Additionally, all of these verbs share the same return dictionary
        after being passed through the model (as really we care about all the frames of the sentence
        together, rather than separately).

        Parameters
        ----------
        json_dict : ``JsonDict``, required.
            JSON that looks like ``{"sentence": "..."}``.

        Returns
        -------
        instances : ``List[Instance]``
            One instance per verb.
        result_dict : ``JsonDict``
            A dictionary containing the words of the sentence and the verbs extracted
            by the Spacy POS tagger. These will be replaced in ``predict_json`` with the
            SRL frame for the verb.
        """
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
        """
        Expects JSON that looks like ``[{"sentence": "..."}, {"sentence": "..."}, ...]``
        and returns JSON that looks like

        .. code-block:: js

            [
                {"words": [...],
                 "verbs": [
                    {"verb": "...", "description": "...", "tags": [...]},
                    ...
                    {"verb": "...", "description": "...", "tags": [...]},
                ]},
                {"words": [...],
                 "verbs": [
                    {"verb": "...", "description": "...", "tags": [...]},
                    ...
                    {"verb": "...", "description": "...", "tags": [...]},
                ]}
            ]
        """
        # For SRL, we have more instances than sentences, but the user specified
        # a batch size with respect to the number of sentences passed, so we respect
        # that here by taking the batch size which we use to be the number of sentences
        # we are given.
        batch_size = len(inputs)
        instances_per_sentence, return_dicts = zip(*[self._sentence_to_srl_instances(json)
                                                     for json in inputs])

        flattened_instances = [instance for sentence_instances in instances_per_sentence
                               for instance in sentence_instances]
        # Make the instances into batches and check the last batch for
        # padded elements as the number of instances might not be perfectly
        # divisible by the batch size.
        batched_instances = group_by_count(flattened_instances, batch_size, None)
        batched_instances[-1] = [instance for instance in batched_instances[-1]
                                 if instance is not None]
        # Run the model on the batches.
        outputs = []
        for batch in batched_instances:
            outputs.extend(self._model.forward_on_instances(batch, cuda_device))

        sentence_index = 0
        for results in return_dicts:
            # We just added the verbs to the list in _sentence_to_srl_instances
            # but we actually want to replace them with their frames, so we
            # reset them here.
            verbs_for_sentence: List[str] = results["verbs"]
            results["verbs"] = []
            # The verbs are in order, but nested as we have multiple sentences.
            # The outputs are already flattened from running through the model,
            # so we just index into this flat list for each verb, updating as we go.
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
        # We just added the verbs to the list in _sentence_to_srl_instances
        # but we actually want to replace them with their frames, so we
        # reset them here.
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
