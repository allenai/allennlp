from typing import Optional, Dict

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

"""
Fairness metrics are based on:
1) Barocas, S.; Hardt, M.; Narayanan, A. 2019. Fairness and machine learning. fairmlbook.org.
1) Zhang, B. H.; Lemoine, B.; and Mitchell, M. 2018. Mitigating unwanted biases with adversarial learning.
In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, 335-340.
2) Hardt, M.; Price, E.; Srebro, N.; et al. 2016. Equality of opportunity in supervised learning.
In Advances in Neural Information Processing Systems, 3315–3323.
3) Beutel, A.; Chen, J.; Zhao, Z.; and Chi, E. H. 2017. Data decisions and theoretical implications when
adversarially learning fair representations. arXiv preprint arXiv:1707.00075.

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
        num_classes : `torch.Tensor`, optional (default = `None`).
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
            if probs.sum() == 0:
                kl_divs[y] = {a: torch.zeros(1)[0] for a in range(num_protected_variables)}
                continue

            C_given_y_dist = Categorical(probs)
            kl_divs[y] = {}
            for a in range(num_protected_variables):
                probs = (
                    predicted_labels[(gold_labels == y) & (protected_variable_labels == a)]
                    .float()
                    .histc(bins=num_classes, min=0, max=num_classes - 1)
                    / predicted_labels.nelement()
                )
                if probs.sum() == 0:
                    kl_divs[y][a] = torch.zeros(1)[0]
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
            if probs.sum() == 0:
                kl_divs[c] = {a: torch.zeros(1)[0] for a in range(num_protected_variables)}
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
                    kl_divs[c][a] = torch.zeros(1)[0]
                    continue

                Y_given_a_and_c_dist = Categorical(probs)
                kl_divs[c][a] = kl_divergence(Y_given_a_and_c_dist, Y_given_c_dist)
        return kl_divs
