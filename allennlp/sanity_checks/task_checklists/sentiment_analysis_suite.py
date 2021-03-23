from typing import Optional
from allennlp.sanity_checks.task_checklists.task_suite import TaskSuite
from checklist.test_suite import TestSuite
from checklist.test_types import MFT

from checklist.editor import Editor
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
        neutral: Optional[int] = 2,
    ):

        self._positive = positive
        self._negative = negative
        self._neutral = neutral

        if not suite:
            suite = TestSuite()
            editor = Editor()

            pos_adj = [
                "good",
                "great",
                "excellent",
                "amazing",
                "extraordinary",
                "beautiful",
                "fantastic",
                "nice",
                "incredible",
                "exceptional",
                "awesome",
                "perfect",
                "fun",
                "happy",
                "adorable",
                "brilliant",
                "exciting",
                "sweet",
                "wonderful",
            ]
            neg_adj = [
                "awful",
                "bad",
                "horrible",
                "weird",
                "rough",
                "lousy",
                "unhappy",
                "average",
                "difficult",
                "poor",
                "sad",
                "frustrating",
                "hard",
                "lame",
                "nasty",
                "annoying",
                "boring",
                "creepy",
                "dreadful",
                "ridiculous",
                "terrible",
                "ugly",
                "unpleasant",
            ]
            neutral_adj = [
                "American",
                "international",
                "commercial",
                "British",
                "private",
                "Italian",
                "Indian",
                "Australian",
                "Israeli",
            ]
            editor.add_lexicon("pos_adj", pos_adj, overwrite=True)
            editor.add_lexicon("neg_adj", neg_adj, overwrite=True)
            editor.add_lexicon("neutral_adj", neutral_adj, overwrite=True)

            pos_verb_present = [
                "like",
                "enjoy",
                "appreciate",
                "love",
                "recommend",
                "admire",
                "value",
                "welcome",
            ]
            neg_verb_present = ["hate", "dislike", "regret", "abhor", "dread", "despise"]
            neutral_verb_present = ["see", "find"]
            pos_verb_past = [
                "liked",
                "enjoyed",
                "appreciated",
                "loved",
                "admired",
                "valued",
                "welcomed",
            ]
            neg_verb_past = ["hated", "disliked", "regretted", "abhorred", "dreaded", "despised"]
            neutral_verb_past = ["saw", "found"]
            editor.add_lexicon("pos_verb_present", pos_verb_present, overwrite=True)
            editor.add_lexicon("neg_verb_present", neg_verb_present, overwrite=True)
            editor.add_lexicon("neutral_verb_present", neutral_verb_present, overwrite=True)
            editor.add_lexicon("pos_verb_past", pos_verb_past, overwrite=True)
            editor.add_lexicon("neg_verb_past", neg_verb_past, overwrite=True)
            editor.add_lexicon("neutral_verb_past", neutral_verb_past, overwrite=True)
            editor.add_lexicon("pos_verb", pos_verb_present + pos_verb_past, overwrite=True)
            editor.add_lexicon("neg_verb", neg_verb_present + neg_verb_past, overwrite=True)
            editor.add_lexicon(
                "neutral_verb", neutral_verb_present + neutral_verb_past, overwrite=True
            )

            suite.add(
                MFT(
                    pos_adj + pos_verb_present + pos_verb_past,
                    labels=self._positive,
                    name="Single Positive Words",
                    capability="Vocabulary",
                    description="Correctly recognizes positive words",
                )
            )

            suite.add(
                MFT(
                    neg_adj + neg_verb_present + neg_verb_past,
                    labels=self._negative,
                    name="Single Negative Words",
                    capability="Vocabulary",
                    description="Correctly recognizes negative words",
                )
            )

            air_noun = [
                "flight",
                "seat",
                "pilot",
                "staff",
                "service",
                "customer service",
                "aircraft",
                "plane",
                "food",
                "cabin crew",
                "company",
                "airline",
                "crew",
            ]
            editor.add_lexicon("air_noun", air_noun)

            template = editor.template(
                "{it} {air_noun} {be} {pos_adj}.",
                it=["The", "This", "That"],
                be=["is", "was"],
                labels=self._positive,
                save=True,
            )
            template += editor.template(
                "{it} {be} {a:pos_adj} {air_noun}.",
                it=["It", "This", "That"],
                be=["is", "was"],
                labels=self._positive,
                save=True,
            )
            template += editor.template(
                "{i} {pos_verb} {the} {air_noun}.",
                i=["I", "We"],
                the=["this", "that", "the"],
                labels=self._positive,
                save=True,
            )
            template += editor.template(
                "{it} {air_noun} {be} {neg_adj}.",
                it=["That", "This", "The"],
                be=["is", "was"],
                labels=self._negative,
                save=True,
            )
            template += editor.template(
                "{it} {be} {a:neg_adj} {air_noun}.",
                it=["It", "This", "That"],
                be=["is", "was"],
                labels=self._negative,
                save=True,
            )
            template += editor.template(
                "{i} {neg_verb} {the} {air_noun}.",
                i=["I", "We"],
                the=["this", "that", "the"],
                labels=self._negative,
                save=True,
            )

            suite.add(
                MFT(**template),
                name="Sentiment-laden words in context",
                capability="Vocabulary",
                description="Use positive and negative verbs and adjectives "
                "with airline nouns such as seats, pilot, flight, etc. "
                'E.g. "This was a bad flight"',
            )

            if self._neutral is not None:
                suite.add(
                    MFT(
                        neutral_adj + neutral_verb_present + neutral_verb_past,
                        name="Single Neutral Words",
                        labels=self._neutral,
                        capability="Vocabulary",
                        description="Correctly recognizes neutral words",
                    )
                )

                template = editor.template(
                    "{it} {air_noun} {be} {neutral_adj}.",
                    it=["That", "This", "The"],
                    be=["is", "was"],
                    save=True,
                )
                template += editor.template(
                    "{it} {be} {a:neutral_adj} {air_noun}.",
                    it=["It", "This", "That"],
                    be=["is", "was"],
                    save=True,
                )
                template += editor.template(
                    "{i} {neutral_verb} {the} {air_noun}.",
                    i=["I", "We"],
                    the=["this", "that", "the"],
                    save=True,
                )
                suite.add(
                    MFT(template.data, labels=self._neutral, templates=template.templates),
                    name="Neutral words in context",
                    capability="Vocabulary",
                    description="Use neutral verbs and adjectives with airline "
                    "nouns such as seats, pilot, flight, etc. "
                    'E.g. "The pilot is American"',
                )

        super().__init__(suite)

    @classmethod
    def _prediction_and_confidence_scores(cls, predictor):
        def preds_and_confs_fn(data):
            labels = []
            confs = []
            data = [{"sentence": sentence} for sentence in data]
            predictions = predictor.predict_batch_json(data)
            for pred in predictions:
                label = pred["probs"].index(max(pred["probs"]))
                labels.append(label)
                confs.append([pred["probs"][0], pred["probs"][1]])
            return np.array(labels), np.array(confs)

        return preds_and_confs_fn
