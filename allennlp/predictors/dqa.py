from overrides import overrides
import json

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('dqa')
class DQAPredictor(Predictor):
    """
    """

    def predict(self, jsonline: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        
        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json(json.loads(jsonline))

    @overrides
    def _json_to_instance(self, paragraph_json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        for paragraph_json in paragraph_json["paragraphs"]:
          paragraph = paragraph_json['context']
          tokenized_paragraph = self._dataset_reader._tokenizer.tokenize(paragraph)
          qas = paragraph_json['qas']
          metadata = {}
          metadata["instance_id"] = [qa['id'] for qa in qas]
          question_text_list = [qa["question"].strip().replace("\n", "") for qa in qas]
          answer_texts_list = [[answer['text'] for answer in qa['answers']] for qa in qas]
          metadata["answer_texts_list"] = answer_texts_list
          metadata["question"] = question_text_list
          span_starts_list = [[answer['answer_start'] for answer in qa['answers']] for qa in qas]
          span_ends_list = []
          for st_list, an_list in zip(span_starts_list, answer_texts_list):
            span_ends = [start + len(answer) for start, answer in zip(st_list, an_list)]
            span_ends_list.append(span_ends)
          yesno_list = [str(qa['yesno']) for qa in qas]
          followup_list = [str(qa['followup']) for qa in qas]
          instance = self._dataset_reader.text_to_instance(question_text_list,
                                                           paragraph,
                                                           span_starts_list,
                                                           span_ends_list,
                                                           tokenized_paragraph,
                                                           yesno_list,
                                                           followup_list,
                                                           metadata)
          return instance
