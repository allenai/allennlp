from typing import Any, Dict, List, Tuple, Union
import torch
from nltk import Tree

from allennlp.common.testing import (
    AllenNlpTestCase,
    global_distributed_metric,
    run_distributed_test,
)
from allennlp.training.metrics import EvalbBracketingScorer


class EvalbBracketingScorerTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        EvalbBracketingScorer.compile_evalb()

    def tearDown(self):
        EvalbBracketingScorer.clean_evalb()
        super().tearDown()

    def test_evalb_correctly_scores_identical_trees(self):
        tree1 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        evalb_scorer = EvalbBracketingScorer()
        evalb_scorer([tree1], [tree2])
        metrics = evalb_scorer.get_metric()
        assert metrics["evalb_recall"] == 1.0
        assert metrics["evalb_precision"] == 1.0
        assert metrics["evalb_f1_measure"] == 1.0

    def test_evalb_correctly_scores_imperfect_trees(self):
        # Change to constiutency label (VP ... )should effect scores, but change to POS
        # tag (NP dog) should have no effect.
        tree1 = Tree.fromstring("(S (VP (D the) (NP dog)) (VP (V chased) (NP (D the) (N cat))))")
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        evalb_scorer = EvalbBracketingScorer()
        evalb_scorer([tree1], [tree2])
        metrics = evalb_scorer.get_metric()
        assert metrics["evalb_recall"] == 0.75
        assert metrics["evalb_precision"] == 0.75
        assert metrics["evalb_f1_measure"] == 0.75

    def test_evalb_correctly_calculates_bracketing_metrics_over_multiple_trees(self):
        tree1 = Tree.fromstring("(S (VP (D the) (NP dog)) (VP (V chased) (NP (D the) (N cat))))")
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        evalb_scorer = EvalbBracketingScorer()
        evalb_scorer([tree1, tree2], [tree2, tree2])
        metrics = evalb_scorer.get_metric()
        assert metrics["evalb_recall"] == 0.875
        assert metrics["evalb_precision"] == 0.875
        assert metrics["evalb_f1_measure"] == 0.875

    def test_evalb_with_terrible_trees_handles_nan_f1(self):
        # If precision and recall are zero, evalb returns nan f1.
        # This checks that we handle the zero division.
        tree1 = Tree.fromstring(
            "(PP (VROOT (PP That) (VROOT (PP could) (VROOT (PP cost) (VROOT (PP him))))) (PP .))"
        )
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        evalb_scorer = EvalbBracketingScorer()
        evalb_scorer([tree1], [tree2])
        metrics = evalb_scorer.get_metric()
        assert metrics["evalb_recall"] == 0.0
        assert metrics["evalb_precision"] == 0.0
        assert metrics["evalb_f1_measure"] == 0.0

    def test_distributed_evalb(self):
        tree1 = Tree.fromstring("(S (VP (D the) (NP dog)) (VP (V chased) (NP (D the) (N cat))))")
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        predicted_trees = [[tree1], [tree2]]
        gold_trees = [[tree2], [tree2]]
        metric_kwargs = {"predicted_trees": predicted_trees, "gold_trees": gold_trees}
        desired_values = {
            "evalb_recall": 0.875,
            "evalb_precision": 0.875,
            "evalb_f1_measure": 0.875,
        }
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            EvalbBracketingScorer(),
            metric_kwargs,
            desired_values,
            exact=True,
        )

    def test_multiple_distributed_runs(self):
        tree1 = Tree.fromstring("(S (VP (D the) (NP dog)) (VP (V chased) (NP (D the) (N cat))))")
        tree2 = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        predicted_trees = [[tree1], [tree2]]
        gold_trees = [[tree2], [tree2]]
        metric_kwargs = {"predicted_trees": predicted_trees, "gold_trees": gold_trees}
        desired_values = {
            "evalb_recall": 0.875,
            "evalb_precision": 0.875,
            "evalb_f1_measure": 0.875,
        }
        run_distributed_test(
            [-1, -1],
            multiple_runs,
            EvalbBracketingScorer(),
            metric_kwargs,
            desired_values,
            exact=False,
        )


def multiple_runs(
    global_rank: int,
    world_size: int,
    gpu_id: Union[int, torch.device],
    metric: EvalbBracketingScorer,
    metric_kwargs: Dict[str, List[Any]],
    desired_values: Dict[str, Any],
    exact: Union[bool, Tuple[float, float]] = True,
):

    kwargs = {}
    # Use the arguments meant for the process with rank `global_rank`.
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]

    for i in range(200):
        metric(**kwargs)

    metric_values = metric.get_metric()

    for key in desired_values:
        assert desired_values[key] == metric_values[key]
