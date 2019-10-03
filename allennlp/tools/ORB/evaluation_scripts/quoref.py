
from typing import Dict, List, Tuple
from allennlp.tools.drop_eval import get_metrics

def get_metric_score(predicted: str, ground_truths: List[str]) -> Tuple[float, float]:
    em_scores = []
    f1_scores = []
    for ground_truth in ground_truths:
        exact_match, f1 = get_metrics(predicted, ground_truth)
        em_scores.append(exact_match)
        f1_scores.append(f1)
    return max(em_scores), max(f1_scores)


def evaluate_quoref(prediction: str, ground_truths: List[str], metrics: Dict[str, Dict]) -> Dict:
    exact_match, f1 = get_metric_score(prediction, ground_truths)
    metrics['quoref']['exact_match'] = metrics['quoref']['exact_match'] + exact_match \
        if 'exact_match' in metrics['quoref'] else exact_match
    metrics['quoref']['f1'] = metrics['quoref']['f1'] + f1 \
        if 'f1' in metrics['quoref'] else f1
    metrics['quoref']['total'] = metrics['quoref']['total'] + 1 \
        if 'total' in metrics['quoref'] else 1
    return metrics
