import sys
import logging
from typing import Type, Optional, Dict, Any, Callable, List, Iterable, Union
from checklist.test_suite import TestSuite
from checklist.editor import Editor
from checklist.test_types import MFT, INV, DIR
from checklist.perturb import Perturb
from allennlp.common.registrable import Registrable
from allennlp.common.file_utils import cached_path
from allennlp.predictors.predictor import Predictor
from allennlp.sanity_checks.task_checklists import utils

logger = logging.getLogger(__name__)


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

    A task suite should contain tests that check general capabilities, including
    but not limited to:

    * Vocabulary + POS : Important words/word types for the task
    * Taxonomy : Synonyms/antonyms, etc.
    * Robustness : To typos, irrelevant changes, etc.
    * NER : Appropriately understanding named entities.
    * Temporal : Understanding the order of events.
    * Negation
    * Coreference
    * Semantic Role Labeling : Understanding roles such as agents and objects.
    * Logic : Ability to handle symmetry, consistency, and conjunctions.
    * Fairness


    # Parameters

    suite: `checklist.test_suite.TestSuite`, optional (default = `None`)
        Pass in an existing test suite.

    add_default_tests: `bool` (default = `False`)
        Whether to add default checklist tests for the task.

    data: `List[Any]`, optional (default = `None`)
        If the data is provided, and `add_default_tests` is `True`,
        tests that perturb the data are also added.

        For instance, if the task is sentiment analysis, and the a
        list of sentences is passed, it will add tests that check
        a model's robustness to typos, etc.
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

    def __init__(
        self,
        suite: Optional[TestSuite] = None,
        add_default_tests: bool = False,
        data: Optional[List[Any]] = None,
        **kwargs,
    ):
        self.suite = suite or TestSuite()

        if add_default_tests:
            self._default_tests(data, **kwargs)

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

        def cap_order(x):
            return self._capabilities.index(x) if x in self._capabilities else 100

        capabilities = sorted(
            set([x["capability"] for x in self.suite.info.values()]), key=cap_order
        )
        print(
            "\n\nThis suite contains {} tests across {} capabilities.".format(
                len(self.suite.tests), len(capabilities)
            )
        )
        print()
        for capability in capabilities:
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
        """
        Saves the suite to a file.
        """
        self.suite.save(suite_file)

    def _default_tests(self, data: Optional[Iterable], num_test_cases=100):
        """
        Derived TaskSuite classes can add any task-specific tests here.
        """
        if data:

            # Robustness

            self._punctuation_test(data, num_test_cases)
            self._typo_test(data, num_test_cases)
            self._contraction_test(data, num_test_cases)

    @classmethod
    def contractions(cls):
        return Perturb.contractions

    @classmethod
    def typos(cls):
        return Perturb.add_typos

    @classmethod
    def punctuation(cls):
        return utils.toggle_punctuation

    def _punctuation_test(self, data, num_test_cases):
        """
        Checks if the model is invariant to presence/absence of punctuation.
        """
        template = Perturb.perturb(data, self.punctuation(), nsamples=num_test_cases)
        # TODO: specify the format_test_case function here.
        test = INV(
            template.data,
            name="Punctuation",
            description="Strip punctuation and / or add '.'",
            capability="Robustness",
        )
        self.add_test(test)

    def _typo_test(self, data, num_test_cases):
        """
        Checks if the model is robust enough to be invariant to simple typos.
        """
        template = Perturb.perturb(data, self.typos(), nsamples=num_test_cases, typos=1)
        test = INV(
            template.data,
            name="Typos",
            capability="Robustness",
            description="Add one typo to input by swapping two adjacent characters",
        )

        self.add_test(test)

        template = Perturb.perturb(data, self.typos(), nsamples=num_test_cases, typos=2)
        test = INV(
            template.data,
            name="2 Typos",
            capability="Robustness",
            description="Add two typos to input by swapping two adjacent characters twice",
        )
        self.add_test(test)

    def _contraction_test(self, data, num_test_cases):
        """
        Checks if the model is invariant to contractions and expansions
        (eg. What is <-> What's) similarly.
        """
        template = Perturb.perturb(data, self.contractions(), nsamples=num_test_cases)
        test = INV(
            template.data,
            name="Contractions",
            capability="Robustness",
            description="Contract or expand contractions, e.g. What is <-> What's",
        )
        self.add_test(test)

    def _setup_editor(self):
        if not hasattr(self, "editor"):
            self.editor = Editor()

    def add_test(self, test: Union[MFT, INV, DIR]):
        """
        Note: `test` needs to be fully specified; with name, capability and description.
        """
        if test.data:  # test data should contain at least one example.
            self.suite.add(test)
        else:
            logger.warning("'{}' was not added, as it contains no examples.".format(test.name))


# We can't decorate `TaskSuite` with `TaskSuite.register()`, because `TaskSuite` hasn't been defined yet.  So we
# put this down here.
TaskSuite.register("from_archive", constructor="constructor")(TaskSuite)
