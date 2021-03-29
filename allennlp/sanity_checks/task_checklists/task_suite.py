import sys
from typing import Type, Optional, Dict, Any, Callable, List
from checklist.test_suite import TestSuite
from allennlp.common.registrable import Registrable
from allennlp.common.file_utils import cached_path
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

    _capabilities = [
        "Vocabulary",
        "Taxonomy",
        "Robustness",
        "NER",
        "Fairness",
        "Temporal",
        "Negation",
        "Coref",
        "SRL",
        "Logic",
    ]

    def __init__(self, suite: Optional[TestSuite] = None, **kwargs):
        self.suite = suite or TestSuite()

    def _prediction_and_confidence_scores(self, predictor: Predictor) -> Callable:
        """
        This makes certain assumptions about the task predictor
        input and output expectations. This should return a function
        that takes the data as input, passes it to the predictor,
        and returns predictions and confidences.
        """
        return NotImplementedError

    def describe(self):
        """
        Gives a description of the test suite.
        """
        capabilities = set([val["capability"] for key, val in self.suite.info.items()])
        print(
            "\n\nThis suite contains {} tests across {} capabilities.".format(
                len(self.suite.tests), len(capabilities)
            )
        )
        print()
        for capability in self._capabilities:
            tests = [
                name for name, test in self.suite.info.items() if test["capability"] == capability
            ]
            if len(tests) > 0:
                print("\n\t{} ({} tests)\n".format(capability, len(tests)))
                for test in tests:
                    description = self.suite.info[test]["description"]
                    num_test_cases = len(self.suite.tests[test].data)
                    about_test = "\t * {} ({} test cases)".format(test, num_test_cases)
                    if description:
                        about_test += " : {}".format(description)
                    print(about_test)

    def summary(self, capabilities=None, file=sys.stdout, **kwargs):
        """
        Prints a summary of the test results.

        # Parameters

        capabilities : `List[str]`, optional (default = `None`)
            If not None, will only show tests with these capabilities.
        **kwargs : `type`
            Will be passed as arguments to each test.summary()
        """
        old_stdout = sys.stdout
        try:
            sys.stdout = file
            self.suite.summary(capabilities=capabilities, **kwargs)
        finally:
            sys.stdout = old_stdout

    def run(
        self,
        predictor: Predictor,
        capabilities: Optional[List[str]] = None,
        max_examples: Optional[int] = None,
    ):
        """
        Runs the predictor on the test suite data.

        # Parameters

        predictor : `Predictor`
            The predictor object.
        capabilities : `List[str]`, optional (default = `None`)
            If not None, will only run tests with these capabilities.
        max_examples : `int`, optional (default = `None`)
            Maximum number of examples to run. If None, all examples will be run.
        """
        preds_and_confs_fn = self._prediction_and_confidence_scores(predictor)
        if preds_and_confs_fn is NotImplementedError:
            raise NotImplementedError(
                "The `_prediction_and_confidence_scores` function needs "
                "to be implemented for the class `{}`".format(self.__class__)
            )
        if not capabilities:
            self.suite.run(preds_and_confs_fn, overwrite=True, n=max_examples)
        else:
            for _, test in self.suite.tests.items():
                if test.capability in capabilities:
                    test.run(preds_and_confs_fn, verbose=True, overwrite=True, n=max_examples)

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
            return suite_class(TestSuite.from_file(cached_path(suite_file)), **extra_args)
        return suite_class(**extra_args)

    def save_suite(self, suite_file: str):
        self.suite.save(suite_file)


# We can't decorate `TaskSuite` with `TaskSuite.register()`, because `TaskSuite` hasn't been defined yet.  So we
# put this down here.
TaskSuite.register("from_archive", constructor="constructor")(TaskSuite)
