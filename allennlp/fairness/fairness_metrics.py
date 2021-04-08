from typing import Optional, Dict, Union

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

"""
Fairness metrics are based on:
1) Barocas, S.; Hardt, M.; and Narayanan, A. 2019. Fairness and machine learning. fairmlbook.org.
2) Zhang, B. H.; Lemoine, B.; and Mitchell, M. 2018. Mitigating unwanted biases with adversarial learning.
In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, 335-340.
3) Hardt, M.; Price, E.; Srebro, N.; et al. 2016. Equality of opportunity in supervised learning.
In Advances in Neural Information Processing Systems, 3315–3323.
4) Beutel, A.; Chen, J.; Zhao, Z.; and Chi, E. H. 2017. Data decisions and theoretical implications when
adversarially learning fair representations. arXiv preprint arXiv:1707.00075.
5) Aka, O.; Burke, K.; Bäuerle, A.; Greer, C.; and Mitchell, M. 2021.
Measuring model biases in the absence of ground truth. arXiv preprint arXiv:2103.03417.

It is provably impossible to satisfy any two of Independence, Separation, and Sufficiency simultaneously,
except in degenerate cases.
"""


@Metric.register("independence")
class Independence(Metric):
    """
    Independence. Assumes integer labels, with
    each item to be classified having a single correct class.
    """

    def __call__(
        self,
        predicted_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> Dict[int, torch.FloatTensor]:
        """
        # Parameters

        predicted_labels : `torch.Tensor`, required.
            A tensor of predicted integer class labels of shape (batch_size, ...). Represented as C.
        protected_variable_labels : `torch.Tensor`, required.
            A tensor of integer protected variable labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as A.
        num_classes : `int`, optional (default = `None`).
            Number of classes. If not supplied, `num_classes` is inferred from `predicted_labels`.

        # Returns

        kl_divs : `Dict[int, torch.FloatTensor]`
            A dictionary mapping each protected variable label a to the KL divergence of P(C | A = a) from P(C).
        """
        predicted_labels, protected_variable_labels = self.detach_tensors(
            predicted_labels, protected_variable_labels
        )

        # Some sanity checks.
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError(
                "protected_variable_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(protected_variable_labels.size())
            )
        if num_classes is not None and (predicted_labels >= num_classes).any():
            raise ConfigurationError(
                "A predicted label contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        if num_classes is None:
            num_classes = predicted_labels.max() + 1
        num_protected_variables = protected_variable_labels.max() + 1

        C_dist = Categorical(
            predicted_labels.float().histc(bins=num_classes, min=0, max=num_classes - 1)
            / predicted_labels.nelement()
        )
        kl_divs: Dict[int, torch.FloatTensor] = {}
        # There currently does not exist a robust loopless way to compute the conditional distributions
        # Assumes num_protected_variables is small
        for a in range(num_protected_variables):
            C_given_a_dist = Categorical(
                predicted_labels[protected_variable_labels == a]
                .float()
                .histc(bins=num_classes, min=0, max=num_classes - 1)
                / predicted_labels.nelement()
            )
            kl_divs[a] = kl_divergence(C_given_a_dist, C_dist)
        return kl_divs


@Metric.register("separation")
class Separation(Metric):
    """
    Separation. Assumes integer labels, with
    each item to be classified having a single correct class.
    """

    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
    ) -> Dict[int, Dict[int, torch.FloatTensor]]:
        """
        # Parameters

        predicted_labels : `torch.Tensor`, required.
            A tensor of predicted integer class labels of shape (batch_size, ...). Represented as C.
        gold_labels : `torch.Tensor`, required.
            A tensor of ground-truth integer class labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as Y.
        protected_variable_labels : `torch.Tensor`, required.
            A tensor of integer protected variable labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as A.

        # Returns

        kl_divs : `Dict[int, Dict[int, torch.FloatTensor]]`
            A dictionary mapping each class label y to a dictionary mapping each protected
            variable label a to the KL divergence of P(C | A = a, Y = y) from P(C | Y = y).

            Note: If a class label is not present in Y conditioned on a protected variable label,
            the expected behavior is that the divergence corresponding to this (class label, protected variable
            label) pair is NaN.
        """
        predicted_labels, gold_labels, protected_variable_labels = self.detach_tensors(
            predicted_labels, gold_labels, protected_variable_labels
        )

        num_classes = gold_labels.max() + 1
        num_protected_variables = protected_variable_labels.max() + 1

        # Some sanity checks.
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError(
                "protected_variable_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(protected_variable_labels.size())
            )
        if predicted_labels.size() != gold_labels.size():
            raise ConfigurationError(
                "gold_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(gold_labels.size())
            )
        if (predicted_labels >= num_classes).any():
            raise ConfigurationError(
                "A predicted label contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        # There currently does not exist a robust loopless way to compute the conditional distributions
        # Assumes num_classes and num_protected_variables are small
        kl_divs: Dict[int, Dict[int, torch.FloatTensor]] = {}
        for y in range(num_classes):
            probs = (
                predicted_labels[gold_labels == y]
                .float()
                .histc(bins=num_classes, min=0, max=num_classes - 1)
                / predicted_labels.nelement()
            )

            C_given_y_dist = Categorical(probs)
            kl_divs[y] = {}
            for a in range(num_protected_variables):
                probs = (
                    predicted_labels[(gold_labels == y) & (protected_variable_labels == a)]
                    .float()
                    .histc(bins=num_classes, min=0, max=num_classes - 1)
                    / predicted_labels.nelement()
                )
                # Implies class label y is not present in Y conditioned on protected variable label a
                if probs.sum() == 0:
                    kl_divs[y][a] = torch.tensor(float("nan"))
                    continue

                C_given_a_and_y_dist = Categorical(probs)
                kl_divs[y][a] = kl_divergence(C_given_a_and_y_dist, C_given_y_dist)
        return kl_divs


@Metric.register("sufficiency")
class Sufficiency(Metric):
    """
    Sufficiency. Assumes integer labels, with
    each item to be classified having a single correct class.
    """

    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
    ) -> Dict[int, Dict[int, torch.FloatTensor]]:
        """
        # Parameters

        predicted_labels : `torch.Tensor`, required.
            A tensor of predicted integer class labels of shape (batch_size, ...). Represented as C.
        gold_labels : `torch.Tensor`, required.
            A tensor of ground-truth integer class labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as Y.
        protected_variable_labels : `torch.Tensor`, required.
            A tensor of integer protected variable labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as A.

        # Returns

        kl_divs : `Dict[int, Dict[int, torch.FloatTensor]]`
            A dictionary mapping each class label c to a dictionary mapping each protected
            variable label a to the KL divergence of P(Y | A = a, C = c) from P(Y | C = c).

            Note: If a possible class label is not present in C, the expected behavior is that
            the divergences corresponding to this class label are NaN. If a possible class label is
            not present in C conditioned on a protected variable label, the expected behavior is that
            the divergence corresponding to this (class label, protected variable label) pair is NaN.
        """
        predicted_labels, gold_labels, protected_variable_labels = self.detach_tensors(
            predicted_labels, gold_labels, protected_variable_labels
        )

        num_classes = gold_labels.max() + 1
        num_protected_variables = protected_variable_labels.max() + 1

        # Some sanity checks.
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError(
                "protected_variable_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(protected_variable_labels.size())
            )
        if predicted_labels.size() != gold_labels.size():
            raise ConfigurationError(
                "gold_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(gold_labels.size())
            )
        if (predicted_labels >= num_classes).any():
            raise ConfigurationError(
                "A predicted label contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        # There currently does not exist a robust loopless way to compute the conditional distributions
        # Assumes num_classes and num_protected_variables are small
        kl_divs: Dict[int, Dict[int, torch.FloatTensor]] = {}
        for c in range(num_classes):
            # It is possible that `c` is not predicted at all,
            # in which case `Y_given_c_dist` is all zeros.
            probs = (
                gold_labels[predicted_labels == c]
                .float()
                .histc(bins=num_classes, min=0, max=num_classes - 1)
                / gold_labels.nelement()
            )
            # Implies class label y is not present in Y conditioned on protected variable label a
            if probs.sum() == 0:
                kl_divs[c] = {a: torch.tensor(float("nan")) for a in range(num_protected_variables)}
                continue

            Y_given_c_dist = Categorical(probs)
            kl_divs[c] = {}
            for a in range(num_protected_variables):
                probs = (
                    gold_labels[(predicted_labels == c) & (protected_variable_labels == a)]
                    .float()
                    .histc(bins=num_classes, min=0, max=num_classes - 1)
                    / gold_labels.nelement()
                )
                if probs.sum() == 0:
                    kl_divs[c][a] = torch.tensor(float("nan"))
                    continue

                Y_given_a_and_c_dist = Categorical(probs)
                kl_divs[c][a] = kl_divergence(Y_given_a_and_c_dist, Y_given_c_dist)
        return kl_divs


@Metric.register("demographic_parity_without_ground_truth")
class DemographicParityWithoutGroundTruth(Metric):
    """
    Demographic parity without ground truth. Assumes integer predictions, with
    each item to be classified having a single correct class.
    From: Aka, O.; Burke, K.; Bäuerle, A.; Greer, C.; and Mitchell, M. 2021.
    Measuring model biases in the absence of ground truth. arXiv preprint arXiv:2103.03417.
    """

    def __init__(self, association_metric: str = "npmixy", gap_type: str = "ova") -> None:
        """
        # Parameters

        association_metric : `str`, optional (default = `npmixy`).
            A generic association metric A(x, y), where x is an identity label and y is any other label.
            Examples include: nPMIxy (`npmixy`), nPMIy (`npmiy`), PMI^2 (`pmisq`), PMI (`pmi`)
            Empirically, nPMIxy and nPMIy are more capable of capturing labels across a range of
            marginal frequencies.
        gap_type : `str`, optional (default = `ova`).
            Either one-vs-all (`ova`) or pairwise (`pairwise`). One-vs-all gap is equivalent to
            A(x, y) - E[A(x', y)], where x' is in the set of all protected variable labels setminus {x}.
            Pairwise gaps are A(x, y) - A(x', y), for all x' in the set of all protected variable labels
            setminus {x}.
        """
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
        num_classes: Optional[int] = None,
    ) -> Dict[int, Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]]:
        """
        # Parameters

        predicted_labels : `torch.Tensor`, required.
            A tensor of predicted integer class labels of shape (batch_size, ...). Represented as Y.
        protected_variable_labels : `torch.Tensor`, required.
            A tensor of integer protected variable labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as X.
        num_classes : `int`, optional (default = `None`).
            Number of classes. If not supplied, `num_classes` is inferred from `predicted_labels`.

        # Returns

        gaps : `Dict[int, Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]]`
            A dictionary mapping each protected variable label x to either:
            1) a tensor of the one-vs-all gaps (where the gap corresponding to prediction
            label i is at index i),
            2) another dictionary mapping protected variable labels x' to a tensor
            of the pairwise gaps (where the gap corresponding to prediction label i is at index i).

            Note: If a possible class label is not present in Y, the expected behavior is that
            the gaps corresponding to this class label are NaN. If a possible (class label,
            protected variable label) pair is not present in the joint of Y and X, the expected
            behavior is that the gap corresponding to this (class label, protected variable label)
            pair is NaN.
        """
        predicted_labels, protected_variable_labels = self.detach_tensors(
            predicted_labels, protected_variable_labels
        )

        # Some sanity checks.
        if predicted_labels.size() != protected_variable_labels.size():
            raise ConfigurationError(
                "protected_variable_labels must be of same size as predicted_labels but "
                "found tensor of shape: {}".format(protected_variable_labels.size())
            )
        if num_classes is not None and (predicted_labels >= num_classes).any():
            raise ConfigurationError(
                "A predicted label contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        if num_classes is None:
            num_classes = predicted_labels.max() + 1
        num_protected_variables = protected_variable_labels.max() + 1
        gaps = {}
        # Assumes num_protected_variables is small
        for x in range(num_protected_variables):
            gaps[x] = self.gap_func(
                x,
                protected_variable_labels.flatten(),
                predicted_labels.flatten(),
                num_classes,
                num_protected_variables,
            )
        return gaps

    def _ova_gap(
        self,
        x: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        num_classes,
        num_protected_variables: int,
    ):
        x_mask = X == x
        joint = torch.zeros(num_classes)
        self._joint(x_mask.long(), Y, out=joint)
        if self.association_metric == "pmisq":
            torch.square_(joint)
        prob_y = torch.zeros(num_classes)
        self._prob_y(Y, out=prob_y)
        pmi_x = torch.log(torch.div(joint, self._prob_x(x_mask) * prob_y))
        if self.association_metric == "npmixy":
            pmi_x.div_(torch.log(joint))
        elif self.association_metric == "npmiy":
            pmi_x.div_(torch.log(prob_y))

        joint = torch.zeros(num_classes)
        self._joint((~x_mask).long(), Y, out=joint)
        if self.association_metric == "pmisq":
            torch.square_(joint)
        pmi_not_x = torch.log(torch.div(joint, self._prob_x(~x_mask) * prob_y)) / (
            num_protected_variables - 1
        )
        if self.association_metric == "npmixy":
            pmi_not_x.div_(torch.log(joint))
        elif self.association_metric == "npmiy":
            pmi_not_x.div_(torch.log(prob_y))

        # Will contain NaN if not all possible class labels are predicted
        # Will contain NaN if not all possible (class label,
        # protected variable label) pairs are predicted
        gap = pmi_x - pmi_not_x
        return torch.where(~gap.isinf(), gap, torch.tensor(float("nan")))

    def _pairwise_gaps(
        self,
        x: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        num_classes,
        num_protected_variables: int,
    ):
        x_mask = X == x
        joint = torch.zeros(num_classes)
        self._joint(x_mask.long(), Y, out=joint)
        if self.association_metric == "pmisq":
            torch.square_(joint)
        prob_y = torch.zeros(num_classes)
        self._prob_y(Y, out=prob_y)
        pmi_x = torch.log(torch.div(joint, self._prob_x(x_mask) * prob_y))
        if self.association_metric == "npmixy":
            pmi_x.div_(torch.log(joint))
        elif self.association_metric == "npmiy":
            pmi_x.div_(torch.log(prob_y))

        pairwise_gaps = {}
        for not_x in range(num_protected_variables):
            not_x_mask = X == not_x
            joint = torch.zeros(num_classes)
            self._joint(not_x_mask.long(), Y, out=joint)
            if self.association_metric == "pmisq":
                torch.square_(joint)
            pmi_not_x = torch.log(torch.div(joint, self._prob_x(not_x_mask) * prob_y))
            if self.association_metric == "npmixy":
                pmi_not_x.div_(torch.log(joint))
            elif self.association_metric == "npmiy":
                pmi_not_x.div_(torch.log(prob_y))

            gap = pmi_x - pmi_not_x
            pairwise_gaps[not_x] = torch.where(~gap.isinf(), gap, torch.tensor(float("nan")))
        return pairwise_gaps

    def _joint(self, x_mask: torch.Tensor, Y: torch.Tensor, out: torch.Tensor):
        counts = torch.zeros_like(out, dtype=x_mask.dtype).scatter_add_(0, Y, x_mask)
        torch.div(counts, Y.nelement(), out=out)

    def _prob_x(self, x_mask: torch.Tensor):
        return x_mask.sum() / x_mask.nelement()

    def _prob_y(self, Y: torch.Tensor, out: torch.Tensor):
        counts = torch.zeros_like(out, dtype=Y.dtype).scatter_add_(0, Y, torch.ones_like(Y))
        torch.div(counts, Y.nelement(), out=out)
