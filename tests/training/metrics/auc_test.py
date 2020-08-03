import pytest
import torch
from sklearn import metrics
from torch.testing import assert_allclose

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    DistributedTestContextManager,
)
from allennlp.training.metrics import Auc


class AucTest(AllenNlpTestCase):
    @multi_device
    def test_auc_computation(self, device: str):
        auc = Auc()
        all_predictions = []
        all_labels = []
        for _ in range(5):
            predictions = torch.randn(8, device=device)
            labels = torch.randint(0, 2, (8,), dtype=torch.long, device=device)

            auc(predictions, labels)

            all_predictions.append(predictions)
            all_labels.append(labels)

        computed_auc_value = auc.get_metric(reset=True)["auc"]

        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
            torch.cat(all_labels, dim=0).cpu().numpy(),
            torch.cat(all_predictions, dim=0).cpu().numpy(),
        )
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_allclose(real_auc_value, computed_auc_value)

        # One more computation to assure reset works.
        predictions = torch.randn(8, device=device)
        labels = torch.randint(0, 2, (8,), dtype=torch.long, device=device)

        auc(predictions, labels)
        computed_auc_value = auc.get_metric(reset=True)["auc"]

        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
            labels.cpu().numpy(), predictions.cpu().numpy()
        )
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_allclose(real_auc_value, computed_auc_value)

    @multi_device
    def test_auc_gold_labels_behaviour(self, device: str):
        # Check that it works with different pos_label
        auc = Auc(positive_label=4)

        predictions = torch.randn(8, device=device)
        labels = torch.randint(3, 5, (8,), dtype=torch.long, device=device)
        # We make sure that the positive label is always present.
        labels[0] = 4
        auc(predictions, labels)
        computed_auc_value = auc.get_metric(reset=True)["auc"]

        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
            labels.cpu().numpy(), predictions.cpu().numpy(), pos_label=4
        )
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_allclose(real_auc_value, computed_auc_value)

        # Check that it errs on getting more than 2 labels.
        with pytest.raises(ConfigurationError) as _:
            labels = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], device=device)
            auc(predictions, labels)

    @multi_device
    def test_auc_with_mask(self, device: str):
        auc = Auc()

        predictions = torch.randn(8, device=device)
        labels = torch.randint(0, 2, (8,), dtype=torch.long, device=device)
        mask = torch.tensor([True, True, True, True, False, False, False, False], device=device)

        auc(predictions, labels, mask)
        computed_auc_value = auc.get_metric(reset=True)["auc"]

        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
            labels[:4].cpu().numpy(), predictions[:4].cpu().numpy()
        )
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_allclose(real_auc_value, computed_auc_value)

    @multi_device
    def test_auc_works_without_calling_metric_at_all(self, device: str):
        auc = Auc()
        auc.get_metric()

    def test_distributed_accuracy(self):
        with DistributedTestContextManager([-1, -1]) as test_this:
            predictions = torch.randn((2, 8))
            labels = torch.randint(3, 5, (2, 8,), dtype=torch.long)
            # We make sure that the positive label is always present.
            labels[:, 0] = 4

            false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
                labels[0].cpu().numpy(), predictions[0].cpu().numpy(), pos_label=4,
            )
            real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
            false_positive_rates, true_positive_rates, _ = metrics.roc_curve(
                labels[1].cpu().numpy(), predictions[1].cpu().numpy(), pos_label=4,
            )

            # NOTE: we return the average.
            real_auc_value += metrics.auc(false_positive_rates, true_positive_rates)
            real_auc_value = real_auc_value / 2

            predictions = [predictions[0], predictions[1]]
            labels = [labels[0], labels[1]]

            metric_kwargs = {"predictions": predictions, "gold_labels": labels}
            desired_values = {"auc": real_auc_value}

            test_this(
                global_distributed_metric,
                Auc(positive_label=4),
                metric_kwargs,
                desired_values,
                exact=False,
            )
