# pylint: disable=no-self-use,invalid-name,protected-access
import torch
from sklearn import metrics
from numpy.testing import assert_almost_equal
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import Auc
from allennlp.common.checks import ConfigurationError


class AucTest(AllenNlpTestCase):
    def test_auc_computation(self):
        auc = Auc()
        all_predictions = []
        all_labels = []
        for _ in range(5):
            predictions = torch.randn(8).float()
            labels = torch.randint(0, 2, (8,)).long()

            auc(predictions, labels)

            all_predictions.append(predictions)
            all_labels.append(labels)

        computed_auc_value = auc.get_metric(reset=True)

        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(torch.cat(all_labels, dim=0).numpy(),
                                                                         torch.cat(all_predictions, dim=0).numpy())
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_almost_equal(real_auc_value, computed_auc_value)

        # One more computation to assure reset works.
        predictions = torch.randn(8).float()
        labels = torch.randint(0, 2, (8,)).long()

        auc(predictions, labels)
        computed_auc_value = auc.get_metric(reset=True)

        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(labels.numpy(),
                                                                         predictions.numpy())
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_almost_equal(real_auc_value, computed_auc_value)

    def test_auc_gold_labels_behaviour(self):
        # Check that it works with different pos_label
        auc = Auc(positive_label=4)

        predictions = torch.randn(8).float()
        labels = torch.randint(3, 5, (8,)).long()

        auc(predictions, labels)
        computed_auc_value = auc.get_metric(reset=True)

        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(labels.numpy(),
                                                                         predictions.numpy(),
                                                                         pos_label=4)
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_almost_equal(real_auc_value, computed_auc_value)

        # Check that it errs on getting more than 2 labels.
        with pytest.raises(ConfigurationError) as _:
            labels = torch.LongTensor([3, 4, 5, 6, 7, 8, 9, 10])
            auc(predictions, labels)

    def test_auc_with_mask(self):
        auc = Auc()

        predictions = torch.randn(8).float()
        labels = torch.randint(0, 2, (8,)).long()
        mask = torch.ByteTensor([1, 1, 1, 1, 0, 0, 0, 0])

        auc(predictions, labels, mask)
        computed_auc_value = auc.get_metric(reset=True)

        false_positive_rates, true_positive_rates, _ = metrics.roc_curve(labels[:4].numpy(),
                                                                         predictions[:4].numpy())
        real_auc_value = metrics.auc(false_positive_rates, true_positive_rates)
        assert_almost_equal(real_auc_value, computed_auc_value)

    def test_auc_works_without_calling_metric_at_all(self):
        auc = Auc()
        auc.get_metric()
