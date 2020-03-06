from typing import Any, Dict, List, Tuple
from collections import Counter

from overrides import overrides
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("conll_coref_scores")
class ConllCorefScores(Metric):
    def __init__(self) -> None:
        self.scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @overrides
    def __call__(
        self,  # type: ignore
        top_spans: torch.Tensor,
        antecedent_indices: torch.Tensor,
        predicted_antecedents: torch.Tensor,
        metadata_list: List[Dict[str, Any]],
    ):
        """
        # Parameters

        top_spans : `torch.Tensor`
            (start, end) indices for all spans kept after span pruning in the model.
            Expected shape: (batch_size, num_spans, 2)
        antecedent_indices : `torch.Tensor`
            For each span, the indices of all allowed antecedents for that span.
            Expected shape: (batch_size, num_spans, num_antecedents)
        predicted_antecedents : `torch.Tensor`
            For each span, this contains the index (into antecedent_indices) of the most likely
            antecedent for that span.
            Expected shape: (batch_size, num_spans)
        metadata_list : `List[Dict[str, Any]]`
            A metadata dictionary for each instance in the batch.  We use the "clusters" key from
            this dictionary, which has the annotated gold coreference clusters for that instance.
        """
        top_spans, antecedent_indices, predicted_antecedents = self.detach_tensors(
            top_spans, antecedent_indices, predicted_antecedents
        )

        # They need to be in CPU because Scorer.ceafe uses a SciPy function.
        top_spans = top_spans.cpu()
        antecedent_indices = antecedent_indices.cpu()
        predicted_antecedents = predicted_antecedents.cpu()

        for i, metadata in enumerate(metadata_list):
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])
            predicted_clusters, mention_to_predicted = self.get_predicted_clusters(
                top_spans[i], antecedent_indices[i], predicted_antecedents[i]
            )
            for scorer in self.scorers:
                scorer.update(
                    predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold
                )

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
        precision, recall, f1_score = tuple(
            sum(metric(e) for e in self.scorers) / len(self.scorers) for metric in metrics
        )
        if reset:
            self.reset()
        return precision, recall, f1_score

    @overrides
    def reset(self):
        self.scorers = [Scorer(metric) for metric in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @staticmethod
    def get_gold_clusters(gold_clusters):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gold_cluster in gold_clusters:
            for mention in gold_cluster:
                mention_to_gold[mention] = gold_cluster
        return gold_clusters, mention_to_gold

    @staticmethod
    def get_predicted_clusters(
        top_spans: torch.Tensor,  # (num_spans, 2)
        antecedent_indices: torch.Tensor,  # (num_spans, num_antecedents)
        predicted_antecedents: torch.Tensor,  # (num_spans,)
    ) -> Tuple[
        List[Tuple[Tuple[int, int], ...]], Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]]
    ]:

        predicted_clusters_to_ids: Dict[Tuple[int, int], int] = {}
        clusters: List[List[Tuple[int, int]]] = []
        for i, predicted_antecedent in enumerate(predicted_antecedents):
            if predicted_antecedent < 0:
                continue

            # Find predicted index in the antecedent spans.
            predicted_index = antecedent_indices[i, predicted_antecedent]
            # Must be a previous span.
            assert i > predicted_index
            antecedent_span: Tuple[int, int] = tuple(  # type: ignore
                top_spans[predicted_index].tolist()
            )

            # Check if we've seen the span before.
            if antecedent_span in predicted_clusters_to_ids.keys():
                predicted_cluster_id: int = predicted_clusters_to_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                clusters.append([antecedent_span])
                predicted_clusters_to_ids[antecedent_span] = predicted_cluster_id

            mention: Tuple[int, int] = tuple(top_spans[i].tolist())  # type: ignore
            clusters[predicted_cluster_id].append(mention)
            predicted_clusters_to_ids[mention] = predicted_cluster_id

        # finalise the spans and clusters.
        final_clusters = [tuple(cluster) for cluster in clusters]
        # Return a mapping of each mention to the cluster containing it.
        mention_to_cluster: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {
            mention: final_clusters[cluster_id]
            for mention, cluster_id in predicted_clusters_to_ids.items()
        }

        return final_clusters, mention_to_cluster


class Scorer:
    """
    Mostly borrowed from <https://github.com/clarkkev/deep-coref/blob/master/evaluation.py>
    """

    def __init__(self, metric):
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0
        self.metric = metric

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == self.ceafe:
            p_num, p_den, r_num, r_den = self.metric(predicted, gold)
        else:
            p_num, p_den = self.metric(predicted, mention_to_gold)
            r_num, r_den = self.metric(gold, mention_to_predicted)
        self.precision_numerator += p_num
        self.precision_denominator += p_den
        self.recall_numerator += r_num
        self.recall_denominator += r_den

    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def get_recall(self):
        if self.recall_denominator == 0:
            return 0
        else:
            return self.recall_numerator / self.recall_denominator

    def get_precision(self):
        if self.precision_denominator == 0:
            return 0
        else:
            return self.precision_numerator / self.precision_denominator

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    @staticmethod
    def b_cubed(clusters, mention_to_gold):
        """
        Averaged per-mention precision and recall.
        <https://pdfs.semanticscholar.org/cfe3/c24695f1c14b78a5b8e95bcbd1c666140fd1.pdf>
        """
        numerator, denominator = 0, 0
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
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def muc(clusters, mention_to_gold):
        """
        Counts the mentions in each predicted cluster which need to be re-allocated in
        order for each predicted cluster to be contained by the respective gold cluster.
        <https://aclweb.org/anthology/M/M95/M95-1005.pdf>
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
    def phi4(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        """
        return (
            2
            * len([mention for mention in gold_clustering if mention in predicted_clustering])
            / (len(gold_clustering) + len(predicted_clustering))
        )

    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        Computes the Constrained Entity-Alignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.

        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        clusters = [cluster for cluster in clusters if len(cluster) != 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = Scorer.phi4(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)
