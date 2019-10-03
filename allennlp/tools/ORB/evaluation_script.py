"""Official evaluation script for ORB.
Usage:
  python evaluation_script.py
        --dataset_file <file_path>
        --prediction_file <file_path>
        --metrics_output_file <file_path>
"""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
from evaluation_scripts.squad_1 import evaluate_squad1
from evaluation_scripts.squad_2 import evaluate_squad2
from evaluation_scripts.narrativeqa import evaluate_narrativeqa
from evaluation_scripts.drop import evaluate_drop
from evaluation_scripts.duorc import evaluate_duorc
from evaluation_scripts.newsqa import evaluate_newsqa
from evaluation_scripts.quoref import evaluate_quoref
from evaluation_scripts.ropes import evaluate_ropes


def read_predictions(json_file):
    return json.load(open(json_file))


def read_labels(jsonl_file):
    qid_answer_map = {}
    with open(jsonl_file) as f:
        for line in f:
            data = json.loads(line)
            for qa_pair in data["qa_pairs"]:
                qid_answer_map[str(qa_pair["qid"])] = {"dataset": qa_pair["dataset"],
                                                       "answers": qa_pair["answers"]}
    return qid_answer_map


def compute_averages(all_metrics):
    for dataset, dataset_metric in all_metrics.items():
        if len(dataset_metric) > 0:
            total = dataset_metric["total"]
            for metric, value in dataset_metric.items():
                if metric != "total":
                    dataset_metric[metric] = value / float(total)
    return all_metrics


def evaluate(answers, predictions):
    metrics = {"drop": {}, "squad1": {}, "squad2": {}, "newsqa": {},
               "quoref": {}, "ropes": {}, "narrativeqa": {}, "duorc": {}}
    for qid, ground_truth_dict in answers.items():
        if qid in predictions:
            predicted_answer = predictions[qid]
            dataset_name = ground_truth_dict["dataset"].lower()
            try:
                evaluate_fn = globals()['evaluate_' + dataset_name]
                metrics = evaluate_fn(predicted_answer, ground_truth_dict["answers"], metrics)
            except KeyError:
                print("Incorrect dataset name at : {0}.".format(dataset_name))
                exit(0)
            except Exception as err:
                print(str(err))

    metrics = compute_averages(metrics)

    return metrics


def process_for_output(metrics):
    processed_metrics = {}
    for dataset, metric_dict in metrics.items():
        for metric_name, metric_value in metric_dict.items():
            if metric_name != "total":
                processed_metrics["{0}_{1}".format(dataset, metric_name)] = metric_value
    return processed_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for ORB')
    parser.add_argument('--dataset_file', type=str, help='Dataset File')
    parser.add_argument('--prediction_file', type=str, help='Prediction File')
    parser.add_argument('--metrics_output_file', type=str, help='Metrics File')
    args = parser.parse_args()

    answers = read_labels(args.dataset_file)
    predictions = read_predictions(args.prediction_file)
    metrics = evaluate(answers, predictions)
    processed_metrics = process_for_output(metrics)
    json.dump(processed_metrics, open(args.metrics_output_file, 'w'), indent=2)
