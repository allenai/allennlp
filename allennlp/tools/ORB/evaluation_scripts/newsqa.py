
from typing import Dict, List
from evaluation_scripts.squad_1 import get_metric_score


def evaluate_newsqa(prediction: str, ground_truths: List[str], metrics: Dict[str, Dict]) -> Dict:
    prediction = prediction[0] if isinstance(prediction, list) else prediction
    exact_match, f1 = get_metric_score(prediction, [truth[0] for truth in ground_truths])
    metrics['newsqa']['exact_match'] = metrics['newsqa']['exact_match'] + exact_match \
        if 'exact_match' in metrics['newsqa'] else exact_match
    metrics['newsqa']['f1'] = metrics['newsqa']['f1'] + f1 if 'f1' in metrics['newsqa'] else f1
    metrics['newsqa']['total'] = metrics['newsqa']['total'] + 1 if 'total' in metrics['newsqa'] else 1
    return metrics
