# pylint: disable=no-self-use,invalid-name,protected-access
from nltk import Tree

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import EvalbBracketingScorer


class EvalbBracketingScorerTest(AllenNlpTestCase):
    def test_evalb_correctly_scores_identical_trees(self):
        tree1 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        print(tree1)
        print(tree2)
        evalb_scorer = EvalbBracketingScorer("scripts/EVALB/")
        evalb_scorer([tree1], [tree2])
        metrics = evalb_scorer.get_metric()
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["f1_measure"] == 1.0
