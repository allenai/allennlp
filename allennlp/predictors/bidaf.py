# pylint: disable=protected-access
from copy import deepcopy
from typing import Dict, List

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import (IndexField, TextField, ListField,
    LabelField, SpanField, SequenceLabelField, SequenceField)


@Predictor.register('machine-comprehension')
class BidafPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """

    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

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
        return self.predict_json({"passage" : passage, "question" : question})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        # For BiDAF
        if 'best_span' in outputs:
            span_start_label = outputs['best_span'][0]
            span_end_label = outputs['best_span'][1]
            passage_field: SequenceField = new_instance['passage'] # type: ignore
            new_instance.add_field('span_start', IndexField(int(span_start_label), passage_field))
            new_instance.add_field('span_end', IndexField(int(span_end_label), passage_field))

        # For NAQANet model. It has the fields: answer_as_passage_spans,
        # answer_as_question_spans, answer_as_add_sub_expressions, answer_as_counts. We need labels for all.
        elif 'answer' in outputs:
            answer_type = outputs['answer']['answer_type']

            # When the problem is a counting problem
            if answer_type == 'count':
                field = ListField([LabelField(int(outputs['answer']['count']), skip_indexing=True)])
                new_instance.add_field('answer_as_counts', field)

            # When the answer is in the passage
            elif answer_type == 'passage_span':
                span = outputs['answer']['spans'][0]

                # Convert character span indices into word span indices
                word_span_start = None
                word_span_end = None
                offsets = new_instance['metadata'].metadata['passage_token_offsets'] # type: ignore
                for index, offset in enumerate(offsets):
                    if offset[0] == span[0]:
                        word_span_start = index
                    if offset[1] == span[1]:
                        word_span_end = index

                passage_field: SequenceField = new_instance['passage']  # type: ignore
                field = ListField([SpanField(word_span_start,
                                             word_span_end,
                                             passage_field)])
                new_instance.add_field('answer_as_passage_spans', field)

            # When the answer is an arithmetic calculation
            elif answer_type == 'arithmetic':
                # The different numbers in the passage that the model encounters
                sequence_labels = outputs['answer']['numbers']

                numbers_as_tokens = [Token(str(number['value']))
                                     for number in outputs['answer']['numbers']]
                # hack copied from https://github.com/
                # allenai/allennlp/blob/master/allennlp/
                # data/dataset_readers/reading_comprehension/drop.py#L232
                numbers_as_tokens.append(Token('0'))
                numbers_in_passage_field = TextField(numbers_as_tokens,
                                                     self._dataset_reader._token_indexers) # type: ignore

                # Based on ``find_valid_add_sub_expressions``, negative signs are 2 instead of -1.
                # The numbers in the passage are given signs, that's what we are labeling here.
                labels = []
                for label in sequence_labels:
                    if label['sign'] == -1:
                        labels.append(2)
                    else:
                        labels.append(label['sign'])
                labels.append(0) # 0 stands for "not included"

                field = ListField([SequenceLabelField(labels, numbers_in_passage_field)])
                new_instance.add_field('answer_as_add_sub_expressions', field)

            # When the answer is in the question
            elif answer_type == 'question_span':
                span = outputs['answer']['spans'][0]

                # Convert character span indices into word span indices
                word_span_start = None
                word_span_end = None
                question_offsets = new_instance['metadata'].metadata['question_token_offsets']  # type: ignore
                for index, offset in question_offsets:
                    if offset[0] == span[0]:
                        word_span_start = index
                    if offset[1] == span[1]:
                        word_span_end = index

                question_field: SequenceField = new_instance['question'] # type: ignore
                field = ListField([SpanField(word_span_start,
                                             word_span_end,
                                             question_field)])
                new_instance.add_field('answer_as_question_spans', field)

        return [new_instance]
