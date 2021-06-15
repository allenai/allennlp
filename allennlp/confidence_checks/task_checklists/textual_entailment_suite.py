from typing import Optional, Tuple, Iterable, Callable, Union
import itertools
import numpy as np
from overrides import overrides
from checklist.test_suite import TestSuite
from checklist.test_types import MFT
from checklist.perturb import Perturb
from allennlp.confidence_checks.task_checklists.task_suite import TaskSuite
from allennlp.confidence_checks.task_checklists import utils


def _wrap_apply_to_each(perturb_fn: Callable, both: bool = False, *args, **kwargs):
    """
    Wraps the perturb function so that it is applied to
    both elements in the (premise, hypothesis) tuple.
    """

    def new_fn(pair, *args, **kwargs):
        premise, hypothesis = pair
        ret = []
        fn_premise = perturb_fn(premise, *args, **kwargs)
        fn_hypothesis = perturb_fn(hypothesis, *args, **kwargs)
        if type(fn_premise) != list:
            fn_premise = [fn_premise]
        if type(fn_hypothesis) != list:
            fn_hypothesis = [fn_hypothesis]
        ret.extend([(x, str(hypothesis)) for x in fn_premise])
        ret.extend([(str(premise), x) for x in fn_hypothesis])
        if both:
            ret.extend([(x, x2) for x, x2 in itertools.product(fn_premise, fn_hypothesis)])

        # The perturb function can return empty strings, if no relevant perturbations
        # can be applied. Eg. if the sentence is "This is a good movie", a perturbation
        # which toggles contractions will have no effect.
        return [x for x in ret if x[0] and x[1]]

    return new_fn


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
        **kwargs,
    ):

        self._entails = entails
        self._contradicts = contradicts
        self._neutral = neutral

        self._premise = premise
        self._hypothesis = hypothesis

        self._probs_key = probs_key

        super().__init__(suite, **kwargs)

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

    @overrides
    def _format_failing_examples(
        self,
        inputs: Tuple,
        pred: int,
        conf: Union[np.array, np.ndarray],
        label: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Formatting function for printing failed test examples.
        """
        labels = {
            self._entails: "Entails",
            self._contradicts: "Contradicts",
            self._neutral: "Neutral",
        }
        ret = "Premise: %s\nHypothesis: %s" % (inputs[0], inputs[1])
        if label is not None:
            ret += "\nOriginal: %s" % labels[label]
        ret += "\nPrediction: Entails (%.1f), Contradicts (%.1f), Neutral (%.1f)" % (
            conf[self._entails],
            conf[self._contradicts],
            conf[self._neutral],
        )

        return ret

    @classmethod
    def contractions(cls):
        return _wrap_apply_to_each(Perturb.contractions, both=True)

    @classmethod
    def typos(cls):
        return _wrap_apply_to_each(Perturb.add_typos, both=False)

    @classmethod
    def punctuation(cls):
        return _wrap_apply_to_each(utils.toggle_punctuation, both=False)

    @overrides
    def _setup_editor(self):
        super()._setup_editor()

        antonyms = [
            ("progressive", "conservative"),
            ("positive", "negative"),
            ("defensive", "offensive"),
            ("rude", "polite"),
            ("optimistic", "pessimistic"),
            ("stupid", "smart"),
            ("negative", "positive"),
            ("unhappy", "happy"),
            ("active", "passive"),
            ("impatient", "patient"),
            ("powerless", "powerful"),
            ("visible", "invisible"),
            ("fat", "thin"),
            ("bad", "good"),
            ("cautious", "brave"),
            ("hopeful", "hopeless"),
            ("insecure", "secure"),
            ("humble", "proud"),
            ("passive", "active"),
            ("dependent", "independent"),
            ("pessimistic", "optimistic"),
            ("irresponsible", "responsible"),
            ("courageous", "fearful"),
        ]

        self.editor.add_lexicon("antonyms", antonyms, overwrite=True)

        comp = [
            "smarter",
            "better",
            "worse",
            "brighter",
            "bigger",
            "louder",
            "longer",
            "larger",
            "smaller",
            "warmer",
            "colder",
            "thicker",
            "lighter",
            "heavier",
        ]

        self.editor.add_lexicon("compare", comp, overwrite=True)

        nouns = [
            "humans",
            "cats",
            "dogs",
            "people",
            "mice",
            "pigs",
            "birds",
            "sheep",
            "cows",
            "rats",
            "chickens",
            "fish",
            "bears",
            "elephants",
            "rabbits",
            "lions",
            "monkeys",
            "snakes",
            "bees",
            "spiders",
            "bats",
            "puppies",
            "dolphins",
            "babies",
            "kittens",
            "children",
            "frogs",
            "ants",
            "butterflies",
            "insects",
            "turtles",
            "trees",
            "ducks",
            "whales",
            "robots",
            "animals",
            "bugs",
            "kids",
            "crabs",
            "carrots",
            "dragons",
            "mosquitoes",
            "cars",
            "sharks",
            "dinosaurs",
            "horses",
            "tigers",
        ]
        self.editor.add_lexicon("nouns", nouns, overwrite=True)

    @overrides
    def _default_tests(self, data: Optional[Iterable[Tuple]], num_test_cases=100):
        super()._default_tests(data, num_test_cases)
        self._setup_editor()
        self._default_vocabulary_tests(data, num_test_cases)
        self._default_ner_tests(data, num_test_cases)
        self._default_temporal_tests(data, num_test_cases)
        self._default_logic_tests(data, num_test_cases)
        self._default_negation_tests(data, num_test_cases)

    def _default_vocabulary_tests(self, data: Optional[Iterable[Tuple]], num_test_cases=100):

        template = self.editor.template(
            (
                "{first_name1} is more {antonyms[0]} than {first_name2}",
                "{first_name2} is more {antonyms[1]} than {first_name1}",
            ),
            remove_duplicates=True,
            nsamples=num_test_cases,
        )

        test = MFT(
            **template,
            labels=self._entails,
            name='"A is more COMP than B" entails "B is more antonym(COMP) than A"',
            capability="Vocabulary",
            description="Eg. A is more active than B implies that B is more passive than A",
        )

        self.add_test(test)

    def _default_logic_tests(self, data: Optional[Iterable[Tuple]], num_test_cases=100):
        template = self.editor.template(
            ("{nouns1} are {compare} than {nouns2}", "{nouns2} are {compare} than {nouns1}"),
            nsamples=num_test_cases,
            remove_duplicates=True,
        )

        test = MFT(
            **template,
            labels=self._contradicts,
            name='"A is COMP than B" contradicts "B is COMP than A"',
            capability="Logic",
            description='Eg. "A is better than B" contradicts "B is better than A"',
        )

        self.add_test(test)

        if data:
            template = Perturb.perturb(
                data, lambda x: (x[0], x[0]), nsamples=num_test_cases, keep_original=False
            )
            template += Perturb.perturb(
                data, lambda x: (x[1], x[1]), nsamples=num_test_cases, keep_original=False
            )

            test = MFT(
                **template,
                labels=self._entails,
                name="A entails A (premise == hypothesis)",
                capability="Logic",
                description="If premise and hypothesis are the same, then premise entails the hypothesis",
            )

            self.add_test(test)

    def _default_negation_tests(self, data: Optional[Iterable[Tuple]], num_test_cases=100):

        template = self.editor.template(
            (
                "{first_name1} is {compare} than {first_name2}",
                "{first_name1} is not {compare} than {first_name2}",
            ),
            nsamples=num_test_cases,
            remove_duplicates=True,
        )

        test = MFT(
            **template,
            labels=self._contradicts,
            name='"A is COMP than B" contradicts "A is not COMP than B"',
            capability="Negation",
            description="Eg. A is better than B contradicts A is not better than C",
        )

        self.add_test(test)

    def _default_ner_tests(self, data: Optional[Iterable[Tuple]], num_test_cases=100):
        template = self.editor.template(
            (
                "{first_name1} is {compare} than {first_name2}",
                "{first_name1} is {compare} than {first_name3}",
            ),
            nsamples=num_test_cases,
            remove_duplicates=True,
        )

        test = MFT(
            **template,
            labels=self._neutral,
            name='"A is COMP than B" gives no information about "A is COMP than C"',
            capability="NER",
            description='Eg. "A is better than B" gives no information about "A is better than C"',
        )

        self.add_test(test)

    def _default_temporal_tests(self, data: Optional[Iterable[Tuple]], num_test_cases=100):
        template = self.editor.template(
            (
                "{first_name} works as {a:profession}",
                "{first_name} used to work as a {profession}",
            ),
            nsamples=num_test_cases,
            remove_duplicates=True,
        )

        template += self.editor.template(
            (
                "{first_name} {last_name} is {a:profession}",
                "{first_name} {last_name} was {a:profession}",
            ),
            nsamples=num_test_cases,
            remove_duplicates=True,
        )

        test = MFT(
            **template,
            labels=self._neutral,
            name='"A works as P" gives no information about "A used to work as P"',
            capability="Temporal",
            description='Eg. "A is a writer" gives no information about "A was a writer"',
        )

        self.add_test(test)

        template = self.editor.template(
            (
                "{first_name} was {a:profession1} before they were {a:profession2}",
                "{first_name} was {a:profession1} after they were {a:profession2}",
            ),
            nsamples=num_test_cases,
            remove_duplicates=True,
        )

        test = MFT(
            **template,
            labels=self._contradicts,
            name="Before != After",
            capability="Temporal",
            description='Eg. "A was a writer before they were a journalist" '
            'contradicts "A was a writer after they were a journalist"',
        )

        self.add_test(test)
