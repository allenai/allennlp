from typing import List, Tuple, Dict
from allennlp.tools.drop_eval import get_metrics

def get_metric_score(predicted: str, ground_truths: List[str]) -> Tuple[float, float]:
    em_scores = []
    f1_scores = []
    for ground_truth in ground_truths:
        exact_match, f1 = get_metrics(predicted, ground_truth)
        em_scores.append(exact_match)
        f1_scores.append(f1)
    return max(em_scores), max(f1_scores)


def evaluate_drop(prediction: str, ground_truths: List[str], metrics: Dict[str, Dict]) -> Dict:
    exact_match, f1 = get_metric_score(prediction, ground_truths)

    metrics['drop']['exact_match'] = metrics['drop']['exact_match'] + exact_match \
        if 'exact_match' in metrics['drop'] else exact_match
    metrics['drop']['f1'] = metrics['drop']['f1'] + f1 if 'f1' in metrics['drop'] else f1
    metrics['drop']['total'] = metrics['drop']['total'] + 1 if 'total' in metrics['drop'] else 1
    return metrics
