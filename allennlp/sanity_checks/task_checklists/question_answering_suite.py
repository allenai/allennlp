from typing import Optional
from allennlp.sanity_checks.task_checklists.task_suite import TaskSuite
from allennlp.sanity_checks.task_checklists import utils
from checklist.perturb import Perturb
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
        **kwargs,
    ):
        self._context_key = context_key
        self._question_key = question_key
        self._answer_key = answer_key

        super().__init__(suite, **kwargs)

    def _prediction_and_confidence_scores(self, predictor):
        def preds_and_confs_fn(data):
            data = [{self._context_key: pair[0], self._question_key: pair[1]} for pair in data]
            predictions = predictor.predict_batch_json(data)
            labels = [pred[self._answer_key] for pred in predictions]
            return labels, np.ones(len(labels))

        return preds_and_confs_fn

    @classmethod
    def contractions(cls):
        def _contractions(x):
            conts = Perturb.contractions(x[1])
            return [(x[0], a) for a in conts]

        return _contractions

    @classmethod
    def typos(cls):
        def question_typo(x):
            return (x[0], Perturb.add_typos(x[1]))

        return question_typo

    @classmethod
    def punctuation(cls):
        def context_punctuation(x):
            return (utils.toggle_punctuation(x[0]), x[1])

        return context_punctuation
