from typing import Optional
from allennlp.sanity_checks.task_checklists.task_suite import TaskSuite
from checklist.test_suite import TestSuite
import numpy as np


@TaskSuite.register("question-answering")
class QuestionAnsweringSuite(TaskSuite):
    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        context_key: str = "context",
        question_key: str = "question",
        answer_key: str = "best_span_str",
    ):
        self._context_key = context_key
        self._question_key = question_key
        self._answer_key = answer_key

        super().__init__(suite)

    def _prediction_and_confidence_scores(self, predictor):
        def preds_and_confs_fn(data):
            data = [{self._context_key: pair[0], self._question_key: pair[1]} for pair in data]
            predictions = predictor.predict_batch_json(data)
            labels = [pred[self._answer_key] for pred in predictions]
            return labels, np.ones(len(labels))

        return preds_and_confs_fn
