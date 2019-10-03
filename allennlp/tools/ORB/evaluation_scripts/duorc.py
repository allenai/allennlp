""" SQuAD evaluation for DuoRC  """

from evaluation_scripts.squad_1 import get_metric_score
from typing import List, Dict


def evaluate_duorc(prediction: str, ground_truths: List[str], metrics: Dict[str, Dict]) -> Dict:
    prediction = prediction[0] if isinstance(prediction, list) else prediction
    exact_match, f1 = get_metric_score(prediction, [truth[0] for truth in ground_truths])
    metrics['duorc']['exact_match'] = metrics['duorc']['exact_match'] + exact_match \
        if 'exact_match' in metrics['duorc'] else exact_match
    metrics['duorc']['f1'] = metrics['duorc']['f1'] + f1 if 'f1' in metrics['duorc'] else f1
    metrics['duorc']['total'] = metrics['duorc']['total'] + 1 if 'total' in metrics['duorc'] else 1
    return metrics
