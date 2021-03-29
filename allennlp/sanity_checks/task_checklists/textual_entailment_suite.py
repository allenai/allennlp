from typing import Optional
from allennlp.sanity_checks.task_checklists.task_suite import TaskSuite
from checklist.test_suite import TestSuite
import numpy as np

@TaskSuite.register("textual-entailment")
class TextualEntailmentSuite(TaskSuite):
    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        entails: int = 0,
        contradicts: int = 1,
        neutral: int = 2,
        premise: str = "premise",
        hypothesis: str = "hypothesis",
        probs_key: str = "probs",
    ):

        self._entails = entails
        self._contradicts = contradicts
        self._neutral = neutral

        self._premise = premise
        self._hypothesis = hypothesis

        self._probs_key = probs_key

        super().__init__(suite)

    def _prediction_and_confidence_scores(self, predictor):
        def preds_and_confs_fn(data):
            labels = []
            confs = []

            data = [{self._premise: pair[0], self._hypothesis: pair[1]} for pair in data]
            predictions = predictor.predict_batch_json(data)
            for pred in predictions:
                label = np.argmax(pred[self._probs_key])
                labels.append(label)
                confs.append(pred[self._probs_key])
            return np.array(labels), np.array(confs)

        return preds_and_confs_fn
