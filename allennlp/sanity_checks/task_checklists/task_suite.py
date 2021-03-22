from typing import Type, Optional, Dict, Any, Callable
from checklist.test_suite import TestSuite
from allennlp.common.registrable import Registrable
from allennlp.predictors.predictor import Predictor


class TaskSuite(Registrable):
    """
    Base class for various task test suites.

    This is a wrapper class around the CheckList toolkit introduced
    in the paper
    [Beyond Accuracy: Behavioral Testing of NLP models with CheckList (Ribeiro et al)]
    (https://api.semanticscholar.org/CorpusID:218551201).

    Task suites are intended to be used as a form of behavioral testing
    for NLP models to check for robustness across several general linguistic
    capabilities; eg. Vocabulary, SRL, Negation, etc.

    An example of the entire checklist process can be found at:
    https://github.com/marcotcr/checklist/blob/master/notebooks/tutorials/
    """

    def __init__(self, suite: Optional[TestSuite] = None, **kwargs):
        self.suite = suite or TestSuite()

    @classmethod
    def _prediction_and_confidence_scores(cls, predictor: Predictor) -> Callable:
        """
        This makes certain assumptions about the task predictor
        input and output expectations. This should return a function
        that takes the data as input, passes it to the predictor,
        and returns predictions and confidences.
        """
        return NotImplementedError

    def run(self, predictor: Predictor):
        """
        Runs the predictor on the test suite data and
        prints a summary of the test results.
        """
        preds_and_confs_fn = self._prediction_and_confidence_scores(predictor)
        if preds_and_confs_fn is NotImplementedError:
            raise NotImplementedError(
                "The `_prediction_and_confidence_scores` function needs "
                "to be implemented for the class `{}`".format(self.__class__)
            )
        self.suite.run(preds_and_confs_fn, overwrite=True)
        self.suite.summary()

    @classmethod
    def constructor(
        cls,
        name: Optional[str] = None,
        suite_file: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> "TaskSuite":
        suite_class: Type[TaskSuite] = (
            TaskSuite.by_name(name) if name is not None else cls  # type: ignore
        )

        if extra_args is None:
            extra_args = {}

        if suite_file is not None:
            return suite_class(TestSuite.from_file(suite_file), **extra_args)
        return suite_class(**extra_args)

    def save_suite(self, suite_file: str):
        self.suite.save(suite_file)


# We can't decorate `TaskSuite` with `TaskSuite.register()`, because `TaskSuite` hasn't been defined yet.  So we
# put this down here.
TaskSuite.register("from_archive", constructor="constructor")(TaskSuite)
