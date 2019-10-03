
from typing import Dict, List
from evaluation_scripts.squad_1 import get_metric_score


def evaluate_ropes(prediction: str, ground_truths: List[str], metrics: Dict[str, Dict]) -> Dict:
    prediction = prediction[0] if isinstance(prediction, list) else prediction
    exact_match, f1 = get_metric_score(prediction, [truth[0] for truth in ground_truths])
    metrics['ropes']['exact_match'] = metrics['ropes']['exact_match'] + exact_match \
        if 'exact_match' in metrics['ropes'] else exact_match
    metrics['ropes']['f1'] = metrics['ropes']['f1'] + f1 if 'f1' in metrics['ropes'] else f1
    metrics['ropes']['total'] = metrics['ropes']['total'] + 1 if 'total' in metrics['ropes'] else 1
    return metrics
