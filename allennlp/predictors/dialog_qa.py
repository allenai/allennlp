import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register("dialog_qa")
class DialogQAPredictor(Predictor):
    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language=language)

    def predict(self, jsonline: str) -> JsonDict:
        """
        Make a dialog-style question answering prediction on the supplied input.
        The supplied input json must contain a list of
        question answer pairs, containing question, answer, yesno, followup, id
        as well as the context (passage).

        Parameters
        ----------
        jsonline : ``str``
            A json line that has the same format as the quac data file.

        Returns
        ----------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json(json.loads(jsonline))

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects json that looks like the original quac data file.
        """
        paragraph_json = json_dict["paragraphs"][0]
        paragraph = paragraph_json["context"]
        tokenized_paragraph = self._tokenizer.tokenize(paragraph)
        qas = paragraph_json["qas"]
        metadata = {}
        metadata["instance_id"] = [qa["id"] for qa in qas]
        question_text_list = [qa["question"].strip().replace("\n", "") for qa in qas]
        answer_texts_list = [[answer["text"] for answer in qa["answers"]] for qa in qas]
        metadata["answer_texts_list"] = answer_texts_list
        metadata["question"] = question_text_list
        span_starts_list = [[answer["answer_start"] for answer in qa["answers"]] for qa in qas]
        span_ends_list = []
        for st_list, an_list in zip(span_starts_list, answer_texts_list):
            span_ends = [start + len(answer) for start, answer in zip(st_list, an_list)]
            span_ends_list.append(span_ends)
        yesno_list = [str(qa["yesno"]) for qa in qas]
        followup_list = [str(qa["followup"]) for qa in qas]
        instance = self._dataset_reader.text_to_instance(
            question_text_list,
            paragraph,
            span_starts_list,
            span_ends_list,
            tokenized_paragraph,
            yesno_list,
            followup_list,
            metadata,
        )
        return instance
