# pylint: disable=no-self-use,invalid-name,protected-access
import os
from nltk import Tree

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import EvalbBracketingScorer


class EvalbBracketingScorerTest(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        os.system("cd ./scripts/EVALB/ && make && cd ../../")

    def tearDown(self):
        os.system("rm scripts/EVALB/evalb")

    def test_evalb_correctly_scores_identical_trees(self):
        tree1 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        evalb_scorer = EvalbBracketingScorer("scripts/EVALB/")
        evalb_scorer([tree1], [tree2])
        metrics = evalb_scorer.get_metric()
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["f1_measure"] == 1.0

    def test_evalb_correctly_scores_imperfect_trees(self):
        # Change to constiutency label (VP ... )should effect scores, but change to POS
        # tag (NP dog) should have no effect.
        tree1 = Tree.fromstring("(S (VP (D the) (NP dog)) (VP (V chased) (NP (D the) (N cat))))")
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        evalb_scorer = EvalbBracketingScorer("scripts/EVALB/")
        evalb_scorer([tree1], [tree2])
        metrics = evalb_scorer.get_metric()
        assert metrics["recall"] == 0.75
        assert metrics["precision"] == 0.75
        assert metrics["f1_measure"] == 0.75
