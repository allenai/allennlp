from typing import Optional
from allennlp.sanity_checks.task_checklists.task_suite import TaskSuite
from checklist.test_suite import TestSuite
import numpy as np


@TaskSuite.register("sentiment-analysis")
class SentimentAnalysisSuite(TaskSuite):
    """
    This suite was built using the checklist process with the editor
    suggestions. Users are encouraged to add/modify as they see fit.

    Note: `editor.suggest(...)` can be slow as it runs a language model.
    """

    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        positive: Optional[int] = 0,
        negative: Optional[int] = 1,
    ):

        self._positive = positive
        self._negative = negative
        super().__init__(suite)

    def _prediction_and_confidence_scores(self, predictor):
        def preds_and_confs_fn(data):
            labels = []
            confs = []
            data = [{"sentence": sentence} for sentence in data]
            predictions = predictor.predict_batch_json(data)
            for pred in predictions:
                label = pred["probs"].index(max(pred["probs"]))
                labels.append(label)
                confs.append([pred["probs"][self._positive], pred["probs"][self._negative]])
            return np.array(labels), np.array(confs)

        return preds_and_confs_fn
