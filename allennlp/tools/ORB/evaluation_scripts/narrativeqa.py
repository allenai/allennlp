""" Evaluation script for NarrativeQA dataset. """

import rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import copy

rouge_l_evaluator = rouge.Rouge(metrics=['rouge-l'], max_n=4, limit_length=True, length_limit=100,
                                length_limit_type='words', apply_avg=True, apply_best=True, alpha=0.5,
                                weight_factor=1.2, stemming=True)


def bleu_1(p, g):
    return sentence_bleu(g, p, weights=(1, 0, 0, 0))


def bleu_4(p, g):
    return sentence_bleu(g, p, weights=(0, 0, 0, 1))


def meteor(p, g):
    return meteor_score(g, p)


def rouge_l(p, g):
    return rouge_l_evaluator.get_scores(p, g)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, tokenize=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if tokenize:
            score = metric_fn(word_tokenize(prediction), [word_tokenize(ground_truth)])
        else:
            score = metric_fn(prediction, [ground_truth])
        scores_for_ground_truths.append(score)
    if isinstance(score, dict) and 'rouge-l' in score:
        max_score = copy.deepcopy(score)
        max_score['rouge-l']['f'] = round(max([score['rouge-l']['f'] for score in scores_for_ground_truths]), 2)
        max_score['rouge-l']['p'] = round(max([score['rouge-l']['p'] for score in scores_for_ground_truths]), 2)
        max_score['rouge-l']['r'] = round(max([score['rouge-l']['r'] for score in scores_for_ground_truths]), 2)
        return max_score
    else:
        return round(max(scores_for_ground_truths), 2)


def get_metric_score(prediction, ground_truths):
    bleu_1_score = metric_max_over_ground_truths(bleu_1, prediction, ground_truths, tokenize=True)
    bleu_4_score = metric_max_over_ground_truths(bleu_4, prediction, ground_truths, tokenize=True)
    meteor_score = metric_max_over_ground_truths(meteor, prediction, ground_truths, tokenize=False)
    rouge_l_score = metric_max_over_ground_truths(rouge_l, prediction, ground_truths, tokenize=False)

    return bleu_1_score, bleu_4_score, meteor_score, rouge_l_score['rouge-l']['f'], \
           rouge_l_score['rouge-l']['p'], rouge_l_score['rouge-l']['r']


def evaluate_narrativeqa(prediction, ground_truths, metrics):
    prediction = prediction[0] if isinstance(prediction, list) else prediction
    ground_truths = [truth[0] for truth in ground_truths]
    bleu_1_score, bleu_4_score, meteor_score, rouge_f, rouge_p, rouge_r = \
        get_metric_score(prediction, ground_truths)

    metrics['narrativeqa']['bleu_1'] = metrics['narrativeqa']['bleu_1'] + bleu_1_score \
        if 'bleu_1' in metrics['narrativeqa'] else bleu_1_score
    metrics['narrativeqa']['bleu_4'] = metrics['narrativeqa']['bleu_4'] + bleu_4_score \
        if 'bleu_4' in metrics['narrativeqa'] else bleu_4_score
    metrics['narrativeqa']['meteor'] = metrics['narrativeqa']['meteor'] + meteor_score \
        if 'meteor' in metrics['narrativeqa'] else meteor_score
    metrics['narrativeqa']['rouge_f'] = metrics['narrativeqa']['rouge_f'] + rouge_f \
        if 'rouge_f' in metrics['narrativeqa'] else rouge_f
    metrics['narrativeqa']['rouge_p'] = metrics['narrativeqa']['rouge_p'] + rouge_p \
        if 'rouge_p' in metrics['narrativeqa'] else rouge_p
    metrics['narrativeqa']['rouge_r'] = metrics['narrativeqa']['rouge_r'] + rouge_r \
        if 'rouge_r' in metrics['narrativeqa'] else rouge_r
    metrics['narrativeqa']['total'] = metrics['narrativeqa']['total'] + 1 \
        if 'total' in metrics['narrativeqa'] else 1

    return metrics
