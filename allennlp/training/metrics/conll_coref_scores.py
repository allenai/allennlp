from typing import Tuple
from collections import Counter
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from overrides import overrides

from allennlp.training.metrics.metric import Metric

@Metric.register("conll_coref_scores")
class ConllCorefScores(Metric):
    def __init__(self) -> None:
        self.scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @overrides
    def __call__(self, top_spans, antecedent_indices, predicted_antecedents, metadata_list):
        top_spans, antecedent_indices, predicted_antecedents = self.unwrap_to_tensors(top_spans,
                                                                                      antecedent_indices,
                                                                                      predicted_antecedents)
        for i, metadata in enumerate(metadata_list):
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])
            predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_spans[i],
                                                                                   antecedent_indices,
                                                                                   predicted_antecedents[i])
            for scorer in self.scorers:
                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
        precision, recall, f1_score = tuple(sum(m(e) for e in self.scorers) / len(self.scorers) for m in metrics)
        if reset:
            self.reset()
        return precision, recall, f1_score

    @overrides
    def reset(self):
        self.scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @staticmethod
    def get_gold_clusters(gold_clusters):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gold_cluster in gold_clusters:
            for mention in gold_cluster:
                mention_to_gold[mention] = gold_cluster
        return gold_clusters, mention_to_gold

    @staticmethod
    def get_predicted_clusters(top_spans, antecedent_indices, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue
            predicted_index = antecedent_indices[i, predicted_antecedent]
            assert i > predicted_index
            predicted_antecedent = tuple(top_spans[predicted_index])
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster
            mention = tuple(top_spans[i])
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster
        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m:predicted_clusters[i] for m, i in mention_to_predicted.items()}
        return predicted_clusters, mention_to_predicted


class Scorer(object):
    """
    Mostly borrowed from <https://github.com/clarkkev/deep-coref/blob/master/evaluation.py>
    """
    def __init__(self, metric):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == self.ceafe:
            p_num, p_den, r_num, r_den = self.metric(predicted, gold)
        else:
            p_num, p_den = self.metric(predicted, mention_to_gold)
            r_num, r_den = self.metric(gold, mention_to_predicted)
        self.p_num += p_num
        self.p_den += p_den
        self.r_num += r_num
        self.r_den += r_den

    def get_f1(self):
        precision = 0 if self.p_den == 0 else self.p_num / float(self.p_den)
        recall = 0 if self.r_den == 0 else self.r_num / float(self.r_den)
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    @staticmethod
    def b_cubed(clusters, mention_to_gold):
        """
        <https://pdfs.semanticscholar.org/cfe3/c24695f1c14b78a5b8e95bcbd1c666140fd1.pdf>
        """
        num, dem = 0, 0
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                if len(cluster2) != 1:
                    correct += count * count
            num += correct / float(len(cluster))
            dem += len(cluster)
        return num, dem

    @staticmethod
    def muc(clusters, mention_to_gold):
        """
        <http://aclweb.org/anthology/M/M95/M95-1005.pdf>
        """
        true_p, all_p = 0, 0
        for cluster in clusters:
            all_p += len(cluster) - 1
            true_p += len(cluster)
            linked = set()
            for mention in cluster:
                if mention in mention_to_gold:
                    linked.add(mention_to_gold[mention])
                else:
                    true_p -= 1
            true_p -= len(linked)
        return true_p, all_p

    @staticmethod
    def phi4(clusters1, clusters2):
        """
        Subroutine for ceafe.
        """
        return 2 * len([m for m in clusters1 if m in clusters2]) / float(len(clusters1) + len(clusters2))

    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        clusters = [c for c in clusters if len(c) != 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = Scorer.phi4(gold_cluster, cluster)
        matching = linear_assignment(-scores)
        similarity = sum(scores[matching[:, 0], matching[:, 1]])
        return similarity, len(clusters), similarity, len(gold_clusters)
