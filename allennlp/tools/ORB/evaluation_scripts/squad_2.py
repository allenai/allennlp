"""Official evaluation script for SQuAD version 2.0.
"""

import collections
import re
import string


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_metric_score(prediction, gold_answers):
    exact_scores = max(compute_exact(a, prediction) for a in gold_answers)
    f1_scores = max(compute_f1(a, prediction) for a in gold_answers)
    return exact_scores, f1_scores


def evaluate_squad2(prediction, ground_truths, metrics):
    prediction = prediction[0] if isinstance(prediction, list) else prediction
    exact_match, f1 = get_metric_score(prediction, [truth[0] for truth in ground_truths])
    metrics['squad2']['exact_match'] = metrics['squad2']['exact_match'] + exact_match \
        if 'exact_match' in metrics['squad2'] else exact_match
    metrics['squad2']['f1'] = metrics['squad2']['f1'] + f1 if 'f1' in metrics['squad2'] else f1
    metrics['squad2']['total'] = metrics['squad2']['total'] + 1 if 'total' in metrics['squad2'] else 1
    return metrics
