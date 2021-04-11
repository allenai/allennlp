"""
Fairness metrics are based on:

1. Barocas, S.; Hardt, M.; and Narayanan, A. 2019. [Fairness and machine learning](https://fairmlbook.org).

2. Zhang, B. H.; Lemoine, B.; and Mitchell, M. 2018. [Mitigating unwanted biases with adversarial learning]
(https://api.semanticscholar.org/CorpusID:9424845).
In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, 335-340.

3. Hardt, M.; Price, E.; Srebro, N.; et al. 2016. [Equality of opportunity in supervised learning]
(https://api.semanticscholar.org/CorpusID:7567061). In Advances in Neural Information Processing Systems,
3315–3323.

4. Beutel, A.; Chen, J.; Zhao, Z.; and Chi, E. H. 2017. [Data decisions and theoretical implications when
adversarially learning fair representations](https://api.semanticscholar.org/CorpusID:24990444).
arXiv preprint arXiv:1707.00075.

5. Aka, O.; Burke, K.; Bäuerle, A.; Greer, C.; and Mitchell, M. 2021.
[Measuring model biases in the absence of ground truth](https://api.semanticscholar.org/CorpusID:232135043).
arXiv preprint arXiv:2103.03417.

It is provably impossible to satisfy any two of Independence, Separation, and Sufficiency simultaneously,
except in degenerate cases.
"""

from typing import Optional, Dict, Union

from overrides import overrides
import torch
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("independence")
class Independence(Metric):
    """
    Independence. Assumes integer labels, with
    each item to be classified having a single correct class.
    """

    def __init__(self, num_classes: int, num_protected_variable_labels: int) -> None:
        """
        # Parameters

        num_classes : `int`.
            Number of classes.
        num_protected_variable_labels : `int`.
            Number of protected variable labels.
        """
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._predicted_label_counts = torch.zeros(num_classes)
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_protected_variable_label = {
            a: torch.zeros(num_classes) for a in range(num_protected_variable_labels)
        }

    def __call__(
        self,
        predicted_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        """
        # Parameters

        predicted_labels : `torch.Tensor`, required.
            A tensor of predicted integer class labels of shape (batch_size, ...). Represented as C.
        protected_variable_labels : `torch.Tensor`, required.
            A tensor of integer protected variable labels of shape (batch_size, ...). It must be the same
            shape as the `predicted_labels` tensor. Represented as A.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_labels`.
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

        if mask is not None:
            predicted_labels = predicted_labels[mask]
            protected_variable_labels = protected_variable_labels[mask]
        else:
            predicted_labels = predicted_labels.flatten()
            protected_variable_labels = protected_variable_labels.flatten()

        _predicted_label_counts = predicted_labels.float().histc(
            bins=self._num_classes, min=0, max=self._num_classes - 1
        )
        _total_predictions = torch.tensor(predicted_labels.nelement())

        _predicted_label_counts_by_protected_variable_label = {}
        for a in range(self._num_protected_variable_labels):
            _predicted_label_counts_by_protected_variable_label[a] = (
                predicted_labels[protected_variable_labels == a]
                .float()
                .histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
            )

        if is_distributed():
            device = torch.device("cuda" if dist.get_backend() == "nccl" else "cpu")

            _predicted_label_counts = _predicted_label_counts.to(device)
            dist.all_reduce(_predicted_label_counts, op=dist.ReduceOp.SUM)

            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)

            for a in range(self._num_protected_variable_labels):
                _predicted_label_counts_by_protected_variable_label[
                    a
                ] = _predicted_label_counts_by_protected_variable_label[a].to(device)
                dist.all_reduce(
                    _predicted_label_counts_by_protected_variable_label[a], op=dist.ReduceOp.SUM
                )

        self._predicted_label_counts += _predicted_label_counts
        self._total_predictions += _total_predictions
        for a in range(self._num_protected_variable_labels):
            self._predicted_label_counts_by_protected_variable_label[
                a
            ] += _predicted_label_counts_by_protected_variable_label[a]

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[int, torch.FloatTensor]:
        """
        # Returns

        kl_divs : `Dict[int, torch.FloatTensor]`
            A dictionary mapping each protected variable label a to the KL divergence of P(C | A = a) from P(C).
            A KL divergence of nearly 0 implies fairness on the basis of Independence.
        """
        kl_divs: Dict[int, torch.FloatTensor] = {}
        if self._total_predictions == 0:
            kl_divs = {
                a: torch.tensor(float("nan")) for a in range(self._num_protected_variable_labels)
            }
            return kl_divs

        C_dist = Categorical(self._predicted_label_counts / self._total_predictions)
        for a in range(self._num_protected_variable_labels):
            C_given_a_dist = Categorical(
                self._predicted_label_counts_by_protected_variable_label[a]
                / self._total_predictions
            )
            kl_divs[a] = kl_divergence(C_given_a_dist, C_dist)
        if reset:
            self.reset()
        return kl_divs

    @overrides
    def reset(self) -> None:
        self._predicted_label_counts = torch.zeros(self._num_classes)
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_protected_variable_label = {
            a: torch.zeros(self._num_classes) for a in range(self._num_protected_variable_labels)
        }


@Metric.register("separation")
class Separation(Metric):
    """
    Separation. Assumes integer labels, with
    each item to be classified having a single correct class.
    """

    def __init__(self, num_classes: int, num_protected_variable_labels: int) -> None:
        """
        # Parameters

        num_classes : `int`.
            Number of classes.
        num_protected_variable_labels : `int`.
            Number of protected variable labels.
        """
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._predicted_label_counts_by_gold_label = {
            y: torch.zeros(num_classes) for y in range(num_classes)
        }
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_gold_label_and_protected_variable_label = {
            y: {a: torch.zeros(num_classes) for a in range(num_protected_variable_labels)}
            for y in range(num_classes)
        }

    @overrides
    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
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
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_labels`.
        """
        predicted_labels, gold_labels, protected_variable_labels, mask = self.detach_tensors(
            predicted_labels, gold_labels, protected_variable_labels, mask
        )

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
        if (gold_labels >= self._num_classes).any():
            raise ConfigurationError(
                "gold_labels contains an id >= {}, "
                "the number of classes.".format(self._num_classes)
            )
        if (protected_variable_labels >= self._num_protected_variable_labels).any():
            raise ConfigurationError(
                "protected_variable_labels contains an id >= {}, "
                "the number of protected variable labels.".format(
                    self._num_protected_variable_labels
                )
            )

        if mask is not None:
            predicted_labels = predicted_labels[mask]
            gold_labels = gold_labels[mask]
            protected_variable_labels = protected_variable_labels[mask]
        else:
            predicted_labels = predicted_labels.flatten()
            gold_labels = gold_labels.flatten()
            protected_variable_labels = protected_variable_labels.flatten()

        _total_predictions = torch.tensor(predicted_labels.nelement())
        _predicted_label_counts_by_gold_label = {}
        _predicted_label_counts_by_gold_label_and_protected_variable_label: Dict[
            int, Dict[int, torch.FloatTensor]
        ] = {}
        for y in range(self._num_classes):
            _predicted_label_counts_by_gold_label[y] = (
                predicted_labels[gold_labels == y]
                .float()
                .histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
            )
            _predicted_label_counts_by_gold_label_and_protected_variable_label[y] = {}
            for a in range(self._num_protected_variable_labels):
                _predicted_label_counts_by_gold_label_and_protected_variable_label[y][a] = (
                    predicted_labels[(gold_labels == y) & (protected_variable_labels == a)]
                    .float()
                    .histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
                )

        if is_distributed():
            device = torch.device("cuda" if dist.get_backend() == "nccl" else "cpu")

            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)

            for y in range(self._num_classes):
                _predicted_label_counts_by_gold_label[y] = _predicted_label_counts_by_gold_label[
                    y
                ].to(device)
                dist.all_reduce(_predicted_label_counts_by_gold_label[y], op=dist.ReduceOp.SUM)

                for a in range(self._num_protected_variable_labels):
                    _predicted_label_counts_by_gold_label_and_protected_variable_label[y][
                        a
                    ] = _predicted_label_counts_by_gold_label_and_protected_variable_label[y][a].to(
                        device
                    )
                    dist.all_reduce(
                        _predicted_label_counts_by_gold_label_and_protected_variable_label[y][a],
                        op=dist.ReduceOp.SUM,
                    )

        self._total_predictions += _total_predictions
        for y in range(self._num_classes):
            self._predicted_label_counts_by_gold_label[y] += _predicted_label_counts_by_gold_label[
                y
            ]
            for a in range(self._num_protected_variable_labels):
                self._predicted_label_counts_by_gold_label_and_protected_variable_label[y][
                    a
                ] += _predicted_label_counts_by_gold_label_and_protected_variable_label[y][a]

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
        """
        # Returns

        kl_divs : `Dict[int, Dict[int, torch.FloatTensor]]`
            A dictionary mapping each class label y to a dictionary mapping each protected
            variable label a to the KL divergence of P(C | A = a, Y = y) from P(C | Y = y).
            A KL divergence of nearly 0 implies fairness on the basis of Separation.

            Note: If a class label is not present in Y conditioned on a protected variable label,
            the expected behavior is that the divergence corresponding to this (class label, protected variable
            label) pair is NaN.
        """
        kl_divs: Dict[int, Dict[int, torch.FloatTensor]] = {}
        if self._total_predictions == 0:
            kl_divs = {
                y: {
                    a: torch.tensor(float("nan"))
                    for a in range(self._num_protected_variable_labels)
                }
                for y in range(self._num_classes)
            }
            return kl_divs

        for y in range(self._num_classes):
            probs = self._predicted_label_counts_by_gold_label[y] / self._total_predictions
            C_given_y_dist = Categorical(probs)
            kl_divs[y] = {}
            for a in range(self._num_protected_variable_labels):
                probs = (
                    self._predicted_label_counts_by_gold_label_and_protected_variable_label[y][a]
                    / self._total_predictions
                )
                # Implies class label y is not present in Y conditioned on protected variable label a
                if probs.sum() == 0:
                    kl_divs[y][a] = torch.tensor(float("nan"))
                    continue
                C_given_a_and_y_dist = Categorical(probs)
                kl_divs[y][a] = kl_divergence(C_given_a_and_y_dist, C_given_y_dist)
        if reset:
            self.reset()
        return kl_divs

    @overrides
    def reset(self) -> None:
        self._predicted_label_counts_by_gold_label = {
            y: torch.zeros(self._num_classes) for y in range(self._num_classes)
        }
        self._total_predictions = torch.tensor(0)
        self._predicted_label_counts_by_gold_label_and_protected_variable_label = {
            y: {
                a: torch.zeros(self._num_classes)
                for a in range(self._num_protected_variable_labels)
            }
            for y in range(self._num_classes)
        }


@Metric.register("sufficiency")
class Sufficiency(Metric):
    """
    Sufficiency. Assumes integer labels, with
    each item to be classified having a single correct class.
    """

    def __init__(self, num_classes: int, num_protected_variable_labels: int) -> None:
        """
        # Parameters

        num_classes : `int`.
            Number of classes.
        num_protected_variable_labels : `int`.
            Number of protected variable labels.
        """
        self._num_classes = num_classes
        self._num_protected_variable_labels = num_protected_variable_labels
        self._gold_label_counts_by_predicted_label = {
            c: torch.zeros(num_classes) for c in range(num_classes)
        }
        self._total_predictions = torch.tensor(0)
        self._gold_label_counts_by_predicted_label_and_protected_variable_label = {
            c: {a: torch.zeros(num_classes) for a in range(num_protected_variable_labels)}
            for c in range(num_classes)
        }

    @overrides
    def __call__(
        self,
        predicted_labels: torch.Tensor,
        gold_labels: torch.Tensor,
        protected_variable_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
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
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of the same shape as `predicted_labels`.
        """
        predicted_labels, gold_labels, protected_variable_labels, mask = self.detach_tensors(
            predicted_labels, gold_labels, protected_variable_labels, mask
        )

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
        if (gold_labels >= self._num_classes).any():
            raise ConfigurationError(
                "gold_labels contains an id >= {}, "
                "the number of classes.".format(self._num_classes)
            )
        if (protected_variable_labels >= self._num_protected_variable_labels).any():
            raise ConfigurationError(
                "protected_variable_labels contains an id >= {}, "
                "the number of protected variable labels.".format(
                    self._num_protected_variable_labels
                )
            )

        if mask is not None:
            predicted_labels = predicted_labels[mask]
            gold_labels = gold_labels[mask]
            protected_variable_labels = protected_variable_labels[mask]
        else:
            predicted_labels = predicted_labels.flatten()
            gold_labels = gold_labels.flatten()
            protected_variable_labels = protected_variable_labels.flatten()

        _total_predictions = torch.tensor(predicted_labels.nelement())
        _gold_label_counts_by_predicted_label = {}
        _gold_label_counts_by_predicted_label_and_protected_variable_label: Dict[
            int, Dict[int, torch.FloatTensor]
        ] = {}
        for c in range(self._num_classes):
            _gold_label_counts_by_predicted_label[c] = (
                gold_labels[predicted_labels == c]
                .float()
                .histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
            )
            _gold_label_counts_by_predicted_label_and_protected_variable_label[c] = {}
            for a in range(self._num_protected_variable_labels):
                _gold_label_counts_by_predicted_label_and_protected_variable_label[c][a] = (
                    gold_labels[(predicted_labels == c) & (protected_variable_labels == a)]
                    .float()
                    .histc(bins=self._num_classes, min=0, max=self._num_classes - 1)
                )

        if is_distributed():
            device = torch.device("cuda" if dist.get_backend() == "nccl" else "cpu")

            _total_predictions = _total_predictions.to(device)
            dist.all_reduce(_total_predictions, op=dist.ReduceOp.SUM)

            for c in range(self._num_classes):
                _gold_label_counts_by_predicted_label[c] = _gold_label_counts_by_predicted_label[
                    c
                ].to(device)
                dist.all_reduce(_gold_label_counts_by_predicted_label[c], op=dist.ReduceOp.SUM)

                for a in range(self._num_protected_variable_labels):
                    _gold_label_counts_by_predicted_label_and_protected_variable_label[c][
                        a
                    ] = _gold_label_counts_by_predicted_label_and_protected_variable_label[c][a].to(
                        device
                    )
                    dist.all_reduce(
                        _gold_label_counts_by_predicted_label_and_protected_variable_label[c][a],
                        op=dist.ReduceOp.SUM,
                    )

        self._total_predictions += _total_predictions
        for c in range(self._num_classes):
            self._gold_label_counts_by_predicted_label[c] += _gold_label_counts_by_predicted_label[
                c
            ]
            for a in range(self._num_protected_variable_labels):
                self._gold_label_counts_by_predicted_label_and_protected_variable_label[c][
                    a
                ] += _gold_label_counts_by_predicted_label_and_protected_variable_label[c][a]

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[int, Dict[int, torch.FloatTensor]]:
        """
        # Returns

        kl_divs : `Dict[int, Dict[int, torch.FloatTensor]]`
            A dictionary mapping each class label c to a dictionary mapping each protected
            variable label a to the KL divergence of P(Y | A = a, C = c) from P(Y | C = c).
            A KL divergence of nearly 0 implies fairness on the basis of Sufficiency.

            Note: If a possible class label is not present in C, the expected behavior is that
            the divergences corresponding to this class label are NaN. If a possible class label is
            not present in C conditioned on a protected variable label, the expected behavior is that
            the divergence corresponding to this (class label, protected variable label) pair is NaN.
        """
        kl_divs: Dict[int, Dict[int, torch.FloatTensor]] = {}
        if self._total_predictions == 0:
            kl_divs = {
                c: {
                    a: torch.tensor(float("nan"))
                    for a in range(self._num_protected_variable_labels)
                }
                for c in range(self._num_classes)
            }
            return kl_divs

        for c in range(self._num_classes):
            # It is possible that `c` is not predicted at all,
            # in which case `probs` is all zeros.
            probs = self._gold_label_counts_by_predicted_label[c] / self._total_predictions
            if probs.sum() == 0:
                kl_divs[c] = {
                    a: torch.tensor(float("nan"))
                    for a in range(self._num_protected_variable_labels)
                }
                continue
            Y_given_c_dist = Categorical(probs)
            kl_divs[c] = {}
            for a in range(self._num_protected_variable_labels):
                probs = (
                    self._gold_label_counts_by_predicted_label_and_protected_variable_label[c][a]
                    / self._total_predictions
                )
                # Implies class label y is not present in Y conditioned on protected variable label a
                if probs.sum() == 0:
                    kl_divs[c][a] = torch.tensor(float("nan"))
                    continue
                Y_given_a_and_c_dist = Categorical(probs)
                kl_divs[c][a] = kl_divergence(Y_given_a_and_c_dist, Y_given_c_dist)
        if reset:
            self.reset()
        return kl_divs

    @overrides
    def reset(self) -> None:
        self._gold_label_counts_by_predicted_label = {
            c: torch.zeros(self._num_classes) for c in range(self._num_classes)
        }
        self._total_predictions = torch.tensor(0)
        self._gold_label_counts_by_predicted_label_and_protected_variable_label = {
            c: {
                a: torch.zeros(self._num_classes)
                for a in range(self._num_protected_variable_labels)
            }
            for c in range(self._num_classes)
        }


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
