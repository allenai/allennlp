#!/usr/bin/python

import json
import sys
import argparse
import string
import numpy as np
import re


# Originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up a bit.
def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def white_space_fix(text):
    return ' '.join(text.split())

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def remove_punc(text):
    if not is_number(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    else:
        return text

def lower(text):
    return text.lower()

# use the number instead of string, if it is one
def norm_number(text):
    if is_number(text):
        return str(float(text))
    else:
        return text

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    sp = ' '.join([white_space_fix(remove_articles(norm_number(remove_punc(lower(tok))))) for tok in s.split()])
    return sp


def answer_to_bags(answer):
    span_bag = set()
    raw_spans = []
    if isinstance(answer, list) or isinstance(answer, tuple):
        raw_spans = answer
    if isinstance(answer, str):
        raw_spans = [answer]
    span_bag = set()
    token_bag = set()
    for raw_span in raw_spans:
        span = normalize_answer(raw_span)
        span_bag.add(span)
        token_bag.update(span.split())
    return span_bag, token_bag


def get_metrics(predicted, gold):
    predicted_bags = answer_to_bags(predicted)
    gold_bags = answer_to_bags(gold)

    exact_match = 1 if predicted_bags[0] == gold_bags[0] else 0

    intersection = len(gold_bags[1].intersection(predicted_bags[1]))
    if len(predicted_bags[1]) == 0:
        precision = 1.0
    else:
        precision = intersection / len(predicted_bags[1])
    if len(gold_bags[1]) == 0:
        recall = 1.0
    else:
        recall = intersection / len(gold_bags[1])
    f1 = (2 * precision * recall) / (precision + recall) if not (precision==0 and recall==0) else 0

    return exact_match, f1


def to_string(answer):
    if answer["number"] != "":
        return tuple([str(answer["number"])]), "number"
    elif len(answer["spans"]) > 0:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    else:
        return tuple(["{0} {1} {2}".format(answer["date"]["day"],
                                           answer["date"]["month"],
                                           answer["date"]["year"])]), "date"


def _run_evaluation(annotations, predicted_answers):
    """
    Evaluation for programatic use.
    """
    exact_match = []
    f1 = []
    # for each type as well
    type_to_em = {}
    type_to_f1 = {}
    for pid, annotation in annotations.items():
        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]
            max_em_score = 0
            max_f1_score = 0
            max_type = None
            if query_id in predicted_answers:
                if "answer" in predicted_answers[query_id]:
                    predicted = predicted_answers[query_id]["answer"]
                else:
                    predicted = predicted_answers[query_id]
            else:
                print("Missing prediction for question: {}".format(query_id))
                predicted = None
            for answer in [qa_pair["answer"]] + qa_pair["validated_answers"]:
                gold_answer, gold_type = to_string(answer)
                em_score, f1_score = get_metrics(predicted, gold_answer)
                if gold_answer[0].strip() != "":
                    max_em_score = max(max_em_score, em_score)
                    max_f1_score = max(max_f1_score, f1_score)
                    if max_em_score == em_score or max_f1_score == f1_score:
                        max_type = gold_type
            exact_match.append(max_em_score)
            f1.append(max_f1_score)
            if max_type not in type_to_em:
                type_to_em[max_type] = []
            type_to_em[max_type].append(max_em_score)
            if max_type not in type_to_f1:
                type_to_f1[max_type] = []
            type_to_f1[max_type].append(max_f1_score)

    global_em = np.mean(exact_match)
    global_f1 = np.mean(f1)
    print("Exact-match accuracy {0:.2f}".format(global_em*100))
    print("F1 score {0:.2f}".format(global_f1*100))
    print("{0:.2f}   &   {1:.2f}".format(global_em*100, global_f1*100))
    print("----")
    total = np.sum([len(v) for v in type_to_em.values()])
    for typ in sorted(type_to_em.keys()):
        print("{0}: {1} ({2:.2f}%)".format(typ, len(type_to_em[typ]), 100.*len(type_to_em[typ])/total))
        print("  Exact-match accuracy {0:.3f}".format(100.*np.mean(type_to_em[typ])))
        print("  F1 score {0:.3f}".format(100.*np.mean(type_to_f1[typ])))
    return global_em, global_f1


def run_evaluation(args):
    predicted_answers = json.load(open(args.prediction_path, encoding='utf-8'))
    annotations = json.load(open(args.gold_path, encoding='utf-8'))
    return _run_evaluation(annotations, predicted_answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate on DROP dataset')
    parser.add_argument("--gold_path", type=str, required=False, default="drop_dataset_test.gold.json",
                    help='location of the gold file')
    parser.add_argument("--prediction_path", type=str, required=True,
                    help='location of the prediction file')
    args = parser.parse_args()
    run_evaluation(args)
