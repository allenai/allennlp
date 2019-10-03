""" Official evaluation script for v1.1 of the SQuAD dataset. """

from collections import Counter
import string
import re


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def get_metric_score(prediction, ground_truths):
    em_scores = []
    f1_scores = []
    for ground_truth in ground_truths:
        em = exact_match_score(prediction, ground_truth)
        f1 = f1_score(prediction, ground_truth)
        em_scores.append(em)
        f1_scores.append(f1)
    return max(em_scores), max(f1_scores)


def evaluate_squad1(prediction, ground_truths, metrics):
    prediction = prediction[0] if isinstance(prediction, list) else prediction
    exact_match, f1 = get_metric_score(prediction, [truth[0] for truth in ground_truths])
    metrics['squad1']['exact_match'] = metrics['squad1']['exact_match'] + exact_match \
        if 'exact_match' in metrics['squad1'] else exact_match
    metrics['squad1']['f1'] = metrics['squad1']['f1'] + f1 if 'f1' in metrics['squad1'] else f1
    metrics['squad1']['total'] = metrics['squad1']['total'] + 1 if 'total' in metrics['squad1'] else 1
    return metrics
