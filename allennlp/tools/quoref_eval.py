"""
This evaluation script relies heavily on the one for DROP (``allennlp/tools/drop_eval.py``). We need a separate
script for Quoref only because the data formats are slightly different.
"""

import json
from typing import Dict, Tuple, List, Any, Optional
import argparse
import numpy as np
from allennlp.tools import drop_eval


def _get_answers_from_data(annotations: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    If the annotations file is in the same format as the original data files, this method can be used to extract a
    dict of query ids and answers.
    """
    answers_dict: Dict[str, List[str]] = {}
    for article_info in annotations["data"]:
        for paragraph_info in article_info["paragraphs"]:
            for qa_pair in paragraph_info["qas"]:
                query_id = qa_pair["id"]
                candidate_answers = [answer["text"] for answer in qa_pair["answers"]]
                answers_dict[query_id] = candidate_answers
    return answers_dict

def evaluate_json(annotations: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float]:
    """
    Takes gold annotations and predicted answers and  evaluates the predictions for each question
    in the gold annotations.  Both JSON dictionaries must have query_id keys, which are used to
    match predictions to gold annotations.

    The ``predicted_answers`` JSON must be a dictionary keyed by query id, where the value is a
    list of strings (or just one string) that is the answer.
    The ``annotations`` are assumed to have either the format of the dev set in the Quoref data release, or the
    same format as the predicted answers file.
    """
    instance_exact_match = []
    instance_f1 = []
    if "data" in annotations:
        # We're looking at annotations in the original data format. Let's extract the answers.
        annotated_answers = _get_answers_from_data(annotations)
    else:
        annotated_answers = annotations
    for query_id, candidate_answers in annotated_answers.items():
        max_em_score = 0.0
        max_f1_score = 0.0
        if query_id in predicted_answers:
            predicted = predicted_answers[query_id]
            gold_answer = tuple(candidate_answers)
            em_score, f1_score = drop_eval.get_metrics(predicted, gold_answer)
            if gold_answer[0].strip() != "":
                max_em_score = max(max_em_score, em_score)
                max_f1_score = max(max_f1_score, f1_score)
        else:
            print("Missing prediction for question: {}".format(query_id))
            max_em_score = 0.0
            max_f1_score = 0.0
        instance_exact_match.append(max_em_score)
        instance_f1.append(max_f1_score)

    global_em = np.mean(instance_exact_match)
    global_f1 = np.mean(instance_f1)
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    return global_em, global_f1


def evaluate_prediction_file(prediction_path: str, gold_path: str,
                             output_path: Optional[str] = None) -> Tuple[float, float]:
    """
    Takes a prediction file and a gold file and evaluates the predictions for each question in the gold file.  Both
    files must be json formatted and must have query_id keys, which are used to match predictions to gold
    annotations. Writes a json with global_em and global_f1 metrics to file at the specified output
    path, unless None is passed as output path.
    """
    predicted_answers = json.load(open(prediction_path, encoding='utf-8'))
    annotations = json.load(open(gold_path, encoding='utf-8'))
    global_em, global_f1 = evaluate_json(annotations, predicted_answers)

    # Output predictions to file if an output path is given
    if output_path is not None:
        output_dict = {"global_em": global_em,
                       "global_f1": global_f1}

        with open(output_path, "w", encoding="utf8") as outfile:
            json.dump(output_dict, outfile)

    return (global_em, global_f1)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Evaluate Quoref predictions')
    parser.add_argument("--gold_path",
                        type=str,
                        required=False,
                        default="quoref-test-v0.1.json",
                        help='location of the gold file')
    parser.add_argument("--prediction_path",
                        type=str,
                        required=False,
                        default="sample_predictions.json",
                        help='location of the prediction file')
    parser.add_argument("--output_path",
                        type=str,
                        required=False,
                        default=None,
                        help='location of the output metrics file')

    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path, args.gold_path, args.output_path)
