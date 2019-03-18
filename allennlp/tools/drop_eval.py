#!/usr/bin/python

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
import json
import argparse
import string
import re

import numpy as np


# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def _remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def _white_space_fix(text: str) -> str:
    return ' '.join(text.split())

EXCLUDE = set(string.punctuation)
def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text

def _lower(text: str) -> str:
    return text.lower()

def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)

def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [_white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
             for token in _tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[Set[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    span_bag: Set[str] = set()
    token_bag = []
    for raw_span in raw_spans:
        span = _normalize_answer(raw_span)
        span_bag.add(span)
        token_bag.append(set(span.split()))
    return span_bag, token_bag


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds a greedy 1-1 alignment
    between them and gets maximum metric values over all the answers
    """
    f1_scores = []
    for gold_index, gold_item in enumerate(gold):
        max_f1 = 0.0
        max_index = None
        best_alignment: Tuple[Set[str], Set[str]] = (set(), set())
        if predicted:
            for pred_index, pred_item in enumerate(predicted):
                current_f1 = _compute_f1(pred_item, gold_item)
                if current_f1 >= max_f1:
                    best_alignment = (gold_item, pred_item)
                    max_f1 = current_f1
                    max_index = pred_index
            match_flag = _match_numbers_if_present(*best_alignment)
            gold[gold_index] = set()
            predicted[max_index] = set()
        else:
            match_flag = False
        if match_flag:
            f1_scores.append(max_f1)
        else:
            f1_scores.append(0.0)
    return f1_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    exact_match = 1.0 if predicted_bags[0] == gold_bags[0] else 0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    elif "date" in answer:
        return tuple(["{0} {1} {2}".format(answer["date"]["day"],
                                           answer["date"]["month"],
                                           answer["date"]["year"])]), "date"
    else:
        raise ValueError(f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}")


def evaluate_json(annotations: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float]:
    """
    Takes gold annotations and predicted answers and  evaluates the predictions for each question
    in the gold annotations.  Both JSON dictionaries must have query_id keys, which are used to
    match predictions to gold annotations (note that these are somewhat deep in the JSON for the
    gold annotations, but must be top-level keys in the predicted answers).

    The ``annotations`` are assumed to have the format of the dev set in the DROP data release.
    The ``predicted_answers`` JSON must be a dictionary keyed by query id, where the value is a string
    (or list of strings) that is the answer.
    """
    instance_exact_match = []
    instance_f1 = []
    # for each type as well
    type_to_em: Dict[str, List[float]] = defaultdict(list)
    type_to_f1: Dict[str, List[float]] = defaultdict(list)
    for query_id, annotation in annotations.items():
        max_em_score = 0.0
        max_f1_score = 0.0
        max_type = None
        if query_id in predicted_answers:
            predicted = predicted_answers[query_id]
            candidate_answers = [annotation["answer"]]
            if "validated_answers" in annotation and annotation["validated_answers"]:
                candidate_answers += annotation["validated_answers"]
            for answer in candidate_answers:
                gold_answer, gold_type = answer_json_to_strings(answer)
                em_score, f1_score = get_metrics(predicted, gold_answer)
                if gold_answer[0].strip() != "":
                    max_em_score = max(max_em_score, em_score)
                    max_f1_score = max(max_f1_score, f1_score)
                    if max_em_score == em_score or max_f1_score == f1_score:
                        max_type = gold_type
        else:
            print("Missing prediction for question: {}".format(query_id))
            _, max_type = answer_json_to_strings(annotation["answer"])
            max_em_score = 0.0
            max_f1_score = 0.0
        instance_exact_match.append(max_em_score)
        instance_f1.append(max_f1_score)
        type_to_em[max_type].append(max_em_score)
        type_to_f1[max_type].append(max_f1_score)

    global_em = np.mean(instance_exact_match)
    global_f1 = np.mean(instance_f1)
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")
    total = np.sum([len(v) for v in type_to_em.values()])
    for typ in sorted(type_to_em.keys()):
        print("{0}: {1} ({2:.2f}%)".format(typ, len(type_to_em[typ]), 100. * len(type_to_em[typ])/total))
        print("  Exact-match accuracy {0:.3f}".format(100. * np.mean(type_to_em[typ])))
        print("  F1 score {0:.3f}".format(100. * np.mean(type_to_f1[typ])))
    return global_em, global_f1


def evaluate_prediction_file(prediction_path: str, gold_path: str) -> Tuple[float, float]:
    """
    Takes a prediction file and a gold file and evaluates the predictions for each question in the
    gold file.  Both files must be json formatted and must have query_id keys, which are used to
    match predictions to gold annotations.  The gold file is assumed to have the format of the dev
    set in the DROP data release.  The prediction file must be a JSON dictionary keyed by query id,
    where the value is either a JSON dictionary with an "answer" key, or just a string (or list of
    strings) that is the answer.
    """
    predicted_answers = json.load(open(prediction_path, encoding='utf-8'))
    annotations = json.load(open(gold_path, encoding='utf-8'))
    return evaluate_json(annotations, predicted_answers)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='evaluate on drop dataset')
    parser.add_argument("--gold_path",
                        type=str,
                        required=False,
                        default="drop_dataset_test.gold.json",
                        help='location of the gold file')
    parser.add_argument("--prediction_path",
                        type=str,
                        required=False,
                        default="sample_predictions.json",
                        help='location of the prediction file')
    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path, args.gold_path)
