"""

A suite of metrics to quantify how much bias is encoded by word embeddings
and determine the effectiveness of bias mitigation.

Bias metrics are based on:

1. Caliskan, A., Bryson, J., & Narayanan, A. (2017). [Semantics derived automatically
from language corpora contain human-like biases](https://api.semanticscholar.org/CorpusID:23163324).
Science, 356, 183 - 186.

2. Dev, S., & Phillips, J.M. (2019). [Attenuating Bias in Word Vectors]
(https://api.semanticscholar.org/CorpusID:59158788). AISTATS.

3. Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020). [On Measuring and Mitigating
Biased Inferences of Word Embeddings](https://api.semanticscholar.org/CorpusID:201670701).
ArXiv, abs/1908.09369.

4. Rathore, A., Dev, S., Phillips, J.M., Srikumar, V., Zheng, Y., Yeh, C.M., Wang, J., Zhang,
W., & Wang, B. (2021). [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
ArXiv, abs/2104.02797.

5. Aka, O.; Burke, K.; Bäuerle, A.; Greer, C.; and Mitchell, M. 2021.
[Measuring model biases in the absence of ground truth](https://api.semanticscholar.org/CorpusID:232135043).
arXiv preprint arXiv:2103.03417.

"""

from typing import Optional, Dict, Union, List

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric


class WordEmbeddingAssociationTest:
    """
    Word Embedding Association Test (WEAT) score measures the unlikelihood there is no
    difference between two sets of target words in terms of their relative similarity
    to two sets of attribute words by computing the probability that a random
    permutation of attribute words would produce the observed (or greater) difference
    in sample means. Analog of Implicit Association Test from psychology for word embeddings.

    Based on: Caliskan, A., Bryson, J., & Narayanan, A. (2017). [Semantics derived automatically
    from language corpora contain human-like biases](https://api.semanticscholar.org/CorpusID:23163324).
    Science, 356, 183 - 186.
    """

    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings1: torch.Tensor,
        attribute_embeddings2: torch.Tensor,
    ) -> torch.FloatTensor:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        target_embeddings1 : `torch.Tensor`, required.
            A tensor of size (target_embeddings_batch_size, ..., dim) containing target word
            embeddings related to a concept group. For example, if the concept is gender,
            target_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc. Represented as X.

        target_embeddings2 : `torch.Tensor`, required.
            A tensor of the same size as target_embeddings1 containing target word
            embeddings related to a different group for the same concept. For example,
            target_embeddings2 could contain embeddings for linguistically feminine words, e.g.
            "woman", "queen", "sister", etc. Represented as Y.

        attribute_embeddings1 : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings1_batch_size, ..., dim) containing attribute word
            embeddings related to a concept group associated with the concept group for target_embeddings1.
            For example, if the concept is professions, attribute_embeddings1 could contain embeddings for
            stereotypically male professions, e.g. "doctor", "banker", "engineer", etc. Represented as A.

        attribute_embeddings2 : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings2_batch_size, ..., dim) containing attribute word
            embeddings related to a concept group associated with the concept group for target_embeddings2.
            For example, if the concept is professions, attribute_embeddings2 could contain embeddings for
            stereotypically female professions, e.g. "nurse", "receptionist", "homemaker", etc. Represented as B.

        !!! Note
            While target_embeddings1 and target_embeddings2 must be the same size, attribute_embeddings1 and
            attribute_embeddings2 need not be the same size.

        # Returns

        weat_score : `torch.FloatTensor`
            The unlikelihood there is no difference between target_embeddings1 and target_embeddings2 in
            terms of their relative similarity to attribute_embeddings1 and attribute_embeddings2.
            Typical values are around [-1, 1], with values closer to 0 indicating less biased associations.

        """

        # Some sanity checks
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise ConfigurationError(
                "target_embeddings1 and target_embeddings2 must have at least two dimensions."
            )
        if attribute_embeddings1.ndim < 2 or attribute_embeddings2.ndim < 2:
            raise ConfigurationError(
                "attribute_embeddings1 and attribute_embeddings2 must have at least two dimensions."
            )
        if target_embeddings1.size() != target_embeddings2.size():
            raise ConfigurationError(
                "target_embeddings1 and target_embeddings2 must be of the same size."
            )
        if attribute_embeddings1.size(dim=-1) != attribute_embeddings2.size(
            dim=-1
        ) or attribute_embeddings1.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise ConfigurationError("All embeddings must have the same dimensionality.")

        target_embeddings1 = target_embeddings1.flatten(end_dim=-2)
        target_embeddings2 = target_embeddings2.flatten(end_dim=-2)
        attribute_embeddings1 = attribute_embeddings1.flatten(end_dim=-2)
        attribute_embeddings2 = attribute_embeddings2.flatten(end_dim=-2)

        # Normalize
        target_embeddings1 = torch.nn.functional.normalize(target_embeddings1, p=2, dim=-1)
        target_embeddings2 = torch.nn.functional.normalize(target_embeddings2, p=2, dim=-1)
        attribute_embeddings1 = torch.nn.functional.normalize(attribute_embeddings1, p=2, dim=-1)
        attribute_embeddings2 = torch.nn.functional.normalize(attribute_embeddings2, p=2, dim=-1)

        # Compute cosine similarities
        X_sim_A = torch.mm(target_embeddings1, attribute_embeddings1.t())
        X_sim_B = torch.mm(target_embeddings1, attribute_embeddings2.t())
        Y_sim_A = torch.mm(target_embeddings2, attribute_embeddings1.t())
        Y_sim_B = torch.mm(target_embeddings2, attribute_embeddings2.t())
        X_union_Y_sim_A = torch.cat([X_sim_A, Y_sim_A])
        X_union_Y_sim_B = torch.cat([X_sim_B, Y_sim_B])

        s_X_A_B = torch.mean(X_sim_A, dim=-1) - torch.mean(X_sim_B, dim=-1)
        s_Y_A_B = torch.mean(Y_sim_A, dim=-1) - torch.mean(Y_sim_B, dim=-1)
        s_X_Y_A_B = torch.mean(s_X_A_B) - torch.mean(s_Y_A_B)
        S_X_union_Y_A_B = torch.mean(X_union_Y_sim_A, dim=-1) - torch.mean(X_union_Y_sim_B, dim=-1)
        return s_X_Y_A_B / torch.std(S_X_union_Y_A_B, unbiased=False)


class EmbeddingCoherenceTest:
    """
    Embedding Coherence Test (ECT) score measures if groups of words
    have stereotypical associations by computing the Spearman Coefficient
    of lists of attribute embeddings sorted based on their similarity to
    target embeddings.

    Based on: Dev, S., & Phillips, J.M. (2019). [Attenuating Bias in Word Vectors]
    (https://api.semanticscholar.org/CorpusID:59158788). AISTATS.
    """

    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings: torch.Tensor,
    ) -> torch.FloatTensor:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        target_embeddings1 : `torch.Tensor`, required.
            A tensor of size (target_embeddings_batch_size, ..., dim) containing target word
            embeddings related to a concept group. For example, if the concept is gender,
            target_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc. Represented as X.

        target_embeddings2 : `torch.Tensor`, required.
            A tensor of the same size as target_embeddings1 containing target word
            embeddings related to a different group for the same concept. For example,
            target_embeddings2 could contain embeddings for linguistically feminine words, e.g.
            "woman", "queen", "sister", etc. Represented as Y.

        attribute_embeddings : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings_batch_size, ..., dim) containing attribute word
            embeddings related to a concept associated with target_embeddings1 and target_embeddings2.
            For example, if the concept is professions, attribute_embeddings could contain embeddings for
            "doctor", "banker", "engineer", etc. Represented as AB.

        # Returns

        ect_score : `torch.FloatTensor`
            The Spearman Coefficient measuring the similarity of lists of attribute embeddings sorted
            based on their similarity to the target embeddings. Ranges from [-1, 1], with values closer
            to 1 indicating less biased associations.

        """
        # Some sanity checks
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise ConfigurationError(
                "target_embeddings1 and target_embeddings2 must have at least two dimensions."
            )
        if attribute_embeddings.ndim < 2:
            raise ConfigurationError("attribute_embeddings must have at least two dimensions.")
        if target_embeddings1.size() != target_embeddings2.size():
            raise ConfigurationError(
                "target_embeddings1 and target_embeddings2 must be of the same size."
            )
        if attribute_embeddings.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise ConfigurationError("All embeddings must have the same dimensionality.")

        mean_target_embedding1 = target_embeddings1.flatten(end_dim=-2).mean(dim=0)
        mean_target_embedding2 = target_embeddings2.flatten(end_dim=-2).mean(dim=0)
        attribute_embeddings = attribute_embeddings.flatten(end_dim=-2)

        # Normalize
        mean_target_embedding1 = torch.nn.functional.normalize(mean_target_embedding1, p=2, dim=-1)
        mean_target_embedding2 = torch.nn.functional.normalize(mean_target_embedding2, p=2, dim=-1)
        attribute_embeddings = torch.nn.functional.normalize(attribute_embeddings, p=2, dim=-1)

        # Compute cosine similarities
        AB_sim_m = torch.matmul(attribute_embeddings, mean_target_embedding1)
        AB_sim_f = torch.matmul(attribute_embeddings, mean_target_embedding2)

        return self.spearman_correlation(AB_sim_m, AB_sim_f)

    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(x.size(0), device=ranks.device)
        return ranks

    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor):
        x_rank = self._get_ranks(x)
        y_rank = self._get_ranks(y)

        n = x.size(0)
        upper = 6 * torch.sum((x_rank - y_rank).pow(2))
        down = n * (n ** 2 - 1.0)
        return 1.0 - (upper / down)


@Metric.register("nli")
class NaturalLanguageInference(Metric):
    """
    Natural language inference scores measure the effect biased associations have on decisions
    made downstream, given neutrally-constructed pairs of sentences differing only in the subject.

    1. Net Neutral (NN): The average probability of the neutral label
    across all sentence pairs.

    2. Fraction Neutral (FN): The fraction of sentence pairs predicted neutral.

    3. Threshold:tau (T:tau): A parameterized measure that reports the fraction
    of examples whose probability of neutral is above tau.

    # Parameters

    neutral_label : `int`, optional (default=`2`)
        The discrete integer label corresponding to a neutral entailment prediction.
    taus : `List[float]`, optional (default=`[0.5, 0.7]`)
        All the taus for which to compute Threshold:tau.

    Based on: Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020). [On Measuring and Mitigating
    Biased Inferences of Word Embeddings](https://api.semanticscholar.org/CorpusID:201670701).
    ArXiv, abs/1908.09369.
    """

    def __init__(self, neutral_label: int = 2, taus: List[float] = [0.5, 0.7]):
        self.neutral_label = neutral_label
        self.taus = taus

        self._nli_probs_sum = 0.0
        self._num_neutral_predictions = 0.0
        self._num_neutral_above_taus = {tau: 0.0 for tau in taus}
        self._total_predictions = 0

    @overrides
    def __call__(self, nli_probabilities: torch.Tensor) -> None:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        nli_probabilities : `torch.Tensor`, required.
            A tensor of size (batch_size, ..., 3) containing natural language inference
            (i.e. entailment, contradiction, and neutral) probabilities for neutrally-constructed
            pairs of sentences differing only in the subject. For example, if the concept is gender,
            nli_probabilities could contain the natural language inference probabilities of:

            - "The driver owns a cabinet." -> "The man owns a cabinet."

            - "The driver owns a cabinet." -> "The woman owns a cabinet."

            - "The doctor eats an apple." -> "The man eats an apple."

            - "The doctor eats an apple." -> "The woman eats an apple."
        """
        nli_probabilities = nli_probabilities.detach()

        # Some sanity checks
        if nli_probabilities.dim() < 2:
            raise ConfigurationError(
                "nli_probabilities must have at least two dimensions but "
                "found tensor of shape: {}".format(nli_probabilities.size())
            )
        if nli_probabilities.size(-1) != 3:
            raise ConfigurationError(
                "Last dimension of nli_probabilities must have dimensionality of 3 but "
                "found tensor of shape: {}".format(nli_probabilities.size())
            )

        _nli_neutral_probs = nli_probabilities[..., self.neutral_label]

        self._nli_probs_sum += dist_reduce_sum(_nli_neutral_probs.sum().item())
        self._num_neutral_predictions += dist_reduce_sum(
            (nli_probabilities.argmax(dim=-1) == self.neutral_label).float().sum().item()
        )
        for tau in self.taus:
            self._num_neutral_above_taus[tau] += dist_reduce_sum(
                (_nli_neutral_probs > tau).float().sum().item()
            )
        self._total_predictions += dist_reduce_sum(_nli_neutral_probs.numel())

    def get_metric(self, reset: bool = False):
        """
        # Returns

        nli_scores : `Dict[str, float]`
            Contains the following keys:

            1. "`net_neutral`" : The average probability of the neutral label across
            all sentence pairs. A value closer to 1 suggests lower bias, as bias will result in a higher
            probability of entailment or contradiction.

            2. "`fraction_neutral`" : The fraction of sentence pairs predicted neutral.
            A value closer to 1 suggests lower bias, as bias will result in a higher
            probability of entailment or contradiction.

            3. "`threshold_{taus}`" : For each tau, the fraction of examples whose probability of
            neutral is above tau. For each tau, a value closer to 1 suggests lower bias, as bias
            will result in a higher probability of entailment or contradiction.

        """
        if self._total_predictions == 0:
            nli_scores = {
                "net_neutral": 0.0,
                "fraction_neutral": 0.0,
                **{"threshold_{}".format(tau): 0.0 for tau in self.taus},
            }
        else:
            nli_scores = {
                "net_neutral": self._nli_probs_sum / self._total_predictions,
                "fraction_neutral": self._num_neutral_predictions / self._total_predictions,
                **{
                    "threshold_{}".format(tau): self._num_neutral_above_taus[tau]
                    / self._total_predictions
                    for tau in self.taus
                },
            }
        if reset:
            self.reset()
        return nli_scores

    @overrides
    def reset(self):
        self._nli_probs_sum = 0.0
        self._num_neutral_predictions = 0.0
        self._num_neutral_above_taus = {tau: 0.0 for tau in self.taus}
        self._total_predictions = 0


@Metric.register("association_without_ground_truth")
class AssociationWithoutGroundTruth(Metric):
    """
    Association without ground truth, from: Aka, O.; Burke, K.; Bäuerle, A.;
    Greer, C.; and Mitchell, M. 2021. Measuring model biases in the absence of ground
    truth. arXiv preprint arXiv:2103.03417.

    # Parameters

    num_classes : `int`
        Number of classes.
    num_protected_variable_labels : `int`
        Number of protected variable labels.
    association_metric : `str`, optional (default = `"npmixy"`).
        A generic association metric A(x, y), where x is an identity label and y is any other label.
        Examples include: nPMIxy (`"npmixy"`), nPMIy (`"npmiy"`), PMI^2 (`"pmisq"`), PMI (`"pmi"`)
        Empirically, nPMIxy and nPMIy are more capable of capturing labels across a range of
        marginal frequencies.
    gap_type : `str`, optional (default = `"ova"`).
        Either one-vs-all (`"ova"`) or pairwise (`"pairwise"`). One-vs-all gap is equivalent to
        A(x, y) - E[A(x', y)], where x' is in the set of all protected variable labels setminus {x}.
        Pairwise gaps are A(x, y) - A(x', y), for all x' in the set of all protected variable labels
        setminus {x}.

    !!! Note
        Assumes integer predictions, with each item to be classified
        having a single correct class.
    """

    def __init__(
        self,
        num_classes: int,
        num_protected_variable_labels: int,
        association_metric: str = "npmixy",
        gap_type: str = "ova",
    ) -> None:
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._joint_counts_by_protected_variable_label = torch.zeros(
            (num_protected_variable_labels, num_classes)
        )
        self._protected_variable_label_counts = torch.zeros(num_protected_variable_labels)
        self._y_counts = torch.zeros(num_classes)
        self._total_predictions = torch.tensor(0)

        self.IMPLEMENTED_ASSOCIATION_METRICS = set(["npmixy", "npmiy", "pmisq", "pmi"])
        if association_metric in self.IMPLEMENTED_ASSOCIATION_METRICS:
            self.association_metric = association_metric
        else:
            raise NotImplementedError(
                f"Association metric {association_metric} has not been implemented!"
            )

        if gap_type == "ova":
            self.gap_func = self._ova_gap
        elif gap_type == "pairwise":
            self.gap_func = self._pairwise_gaps
        else:
            raise NotImplementedError(f"Gap type {gap_type} has not been implemented!")

    def __call__(
        self,
        predicted_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        """
        # Parameters

        predicted_labels : `torch.Tensor`, required.
            A tensor of predicted integer class labels of shape (batch_size, ...). Represented as Y.
        protected_variable_labels : `torch.Tensor`, required.
            A tensor of integer protected variable labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as X.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_labels`.

        !!! Note
            All tensors are expected to be on the same device.
        """
        predicted_labels, protected_variable_labels, mask = self.detach_tensors(
            predicted_labels, protected_variable_labels, mask
        )

        # Some sanity checks.
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError(
                "protected_variable_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(protected_variable_labels.size())
            )
        if mask is not None and predicted_labels.size() != mask.size():
            raise ConfigurationError(
                "mask must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(mask.size())
            )
        if (predicted_labels >= self._num_classes).any():
            raise ConfigurationError(
                "predicted_labels contains an id >= {}, "
                "the number of classes.".format(self._num_classes)
            )
        if (protected_variable_labels >= self._num_protected_variable_labels).any():
            raise ConfigurationError(
                "protected_variable_labels contains an id >= {}, "
                "the number of protected variable labels.".format(
                    self._num_protected_variable_labels
                )
            )

        device = predicted_labels.device
        self._joint_counts_by_protected_variable_label = (
            self._joint_counts_by_protected_variable_label.to(device)
        )
        self._protected_variable_label_counts = self._protected_variable_label_counts.to(device)
        self._y_counts = self._y_counts.to(device)
        self._total_predictions = self._total_predictions.to(device)

        if mask is not None:
            predicted_labels = predicted_labels[mask]
            protected_variable_labels = protected_variable_labels[mask]
        else:
            predicted_labels = predicted_labels.flatten()
            protected_variable_labels = protected_variable_labels.flatten()

        _total_predictions = torch.tensor(predicted_labels.nelement()).to(device)
        _y_counts = torch.zeros(self._num_classes).to(device)
        _y_counts = torch.zeros_like(_y_counts, dtype=predicted_labels.dtype).scatter_add_(
            0, predicted_labels, torch.ones_like(predicted_labels)
        )

        _joint_counts_by_protected_variable_label = torch.zeros(
            (self._num_protected_variable_labels, self._num_classes)
        ).to(device)
        _protected_variable_label_counts = torch.zeros(self._num_protected_variable_labels).to(
            device
        )
        for x in range(self._num_protected_variable_labels):
            x_mask = (protected_variable_labels == x).long()

            _joint_counts_by_protected_variable_label[x] = torch.zeros(self._num_classes).to(device)
            _joint_counts_by_protected_variable_label[x] = torch.zeros_like(
                _joint_counts_by_protected_variable_label[x], dtype=x_mask.dtype
            ).scatter_add_(0, predicted_labels, x_mask)

            _protected_variable_label_counts[x] = torch.tensor(x_mask.sum()).to(device)

        if is_distributed():
            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)

            _y_counts = _y_counts.to(device)
            dist.all_reduce(_y_counts, op=dist.ReduceOp.SUM)

            _joint_counts_by_protected_variable_label = (
                _joint_counts_by_protected_variable_label.to(device)
            )
            dist.all_reduce(_joint_counts_by_protected_variable_label, op=dist.ReduceOp.SUM)

            _protected_variable_label_counts = _protected_variable_label_counts.to(device)
            dist.all_reduce(_protected_variable_label_counts, op=dist.ReduceOp.SUM)

        self._total_predictions += _total_predictions
        self._y_counts += _y_counts
        self._joint_counts_by_protected_variable_label += _joint_counts_by_protected_variable_label
        self._protected_variable_label_counts += _protected_variable_label_counts

    @overrides
    def get_metric(
        self, reset: bool = False
    ) -> Dict[int, Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]]:
        """
        # Returns

        gaps : `Dict[int, Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]]`
            A dictionary mapping each protected variable label x to either:

            1. a tensor of the one-vs-all gaps (where the gap corresponding to prediction
            label i is at index i),

            2. another dictionary mapping protected variable labels x' to a tensor
            of the pairwise gaps (where the gap corresponding to prediction label i is at index i).
            A gap of nearly 0 implies fairness on the basis of Association in the Absence of Ground Truth.

        !!! Note
            If a possible class label is not present in Y, the expected behavior is that
            the gaps corresponding to this class label are NaN. If a possible (class label,
            protected variable label) pair is not present in the joint of Y and X, the expected
            behavior is that the gap corresponding to this (class label, protected variable label)
            pair is NaN.
        """
        gaps = {}
        for x in range(self._num_protected_variable_labels):
            gaps[x] = self.gap_func(x)
        if reset:
            self.reset()
        return gaps

    @overrides
    def reset(self) -> None:
        self._joint_counts_by_protected_variable_label = torch.zeros(
            (self._num_protected_variable_labels, self._num_classes)
        )
        self._protected_variable_label_counts = torch.zeros(self._num_protected_variable_labels)
        self._y_counts = torch.zeros(self._num_classes)
        self._total_predictions = torch.tensor(0)

    def _ova_gap(self, x: int):
        device = self._y_counts.device
        pmi_terms = self._all_pmi_terms()
        pmi_not_x = torch.sum(
            pmi_terms[torch.arange(self._num_protected_variable_labels, device=device) != x], dim=0
        )
        pmi_not_x /= self._num_protected_variable_labels - 1

        # Will contain NaN if not all possible class labels are predicted
        # Will contain NaN if not all possible (class label,
        # protected variable label) pairs are predicted
        gap = pmi_terms[x] - pmi_not_x
        return torch.where(~gap.isinf(), gap, torch.tensor(float("nan")).to(device))

    def _pairwise_gaps(self, x: int):
        device = self._y_counts.device
        pmi_terms = self._all_pmi_terms()
        pairwise_gaps = {}
        for not_x in range(self._num_protected_variable_labels):
            gap = pmi_terms[x] - pmi_terms[not_x]
            pairwise_gaps[not_x] = torch.where(
                ~gap.isinf(), gap, torch.tensor(float("nan")).to(device)
            )
        return pairwise_gaps

    def _all_pmi_terms(self) -> Dict[int, torch.Tensor]:
        if self._total_predictions == 0:
            return torch.full(
                (self._num_protected_variable_labels, self._num_classes), float("nan")
            )

        device = self._y_counts.device
        prob_y = torch.zeros(self._num_classes).to(device)
        torch.div(self._y_counts, self._total_predictions, out=prob_y)

        joint = torch.zeros((self._num_protected_variable_labels, self._num_classes)).to(device)
        torch.div(
            self._joint_counts_by_protected_variable_label,
            self._total_predictions,
            out=joint,
        )
        if self.association_metric == "pmisq":
            torch.square_(joint)

        pmi_terms = torch.log(
            torch.div(
                joint,
                (self._protected_variable_label_counts / self._total_predictions).unsqueeze(-1)
                * prob_y,
            )
        )
        if self.association_metric == "npmixy":
            pmi_terms.div_(torch.log(joint))
        elif self.association_metric == "npmiy":
            pmi_terms.div_(torch.log(prob_y))

        return pmi_terms
