import math

import torch
from torch.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.training.metrics import SpearmanCorrelation


def spearman_formula(predictions, labels, mask=None):
    """
    This function is spearman formula from:
        https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    """
    if mask is not None:
        predictions = predictions * mask
        labels = labels * mask

    # if all number of a set is same, return np.nan
    if len(torch.unique(predictions)) == 1 or len(torch.unique(labels)) == 1:
        return float("NaN")

    len_pre = len(predictions)

    predictions = [(k, v) for k, v in enumerate(predictions)]
    predictions.sort(key=lambda x: x[1], reverse=True)
    predictions = [(k, v) for k, v in enumerate(predictions)]
    predictions.sort(key=lambda x: x[1][0])

    labels = [(k, v) for k, v in enumerate(labels)]
    labels.sort(key=lambda x: x[1], reverse=True)
    labels = [(k, v) for k, v in enumerate(labels)]
    labels.sort(key=lambda x: x[1][0])

    total = 0
    for i in range(len_pre):
        total += (predictions[i][0] - labels[i][0]) ** 2
    expected_spearman_correlation = 1 - 6 * total / (len_pre * (len_pre ** 2 - 1))

    return expected_spearman_correlation


class SpearmanCorrelationTest(AllenNlpTestCase):
    @multi_device
    def test_unmasked_computation(self, device: str):
        spearman_correlation = SpearmanCorrelation()
        batch_size = 10
        num_labels = 10
        predictions1 = torch.randn(batch_size, num_labels, device=device)
        labels1 = 0.5 * predictions1 + torch.randn(batch_size, num_labels, device=device)

        predictions2 = torch.randn(1, device=device).repeat(num_labels)
        predictions2 = predictions2.unsqueeze(0).expand(batch_size, -1)
        labels2 = torch.randn(1, device=device).expand(num_labels)
        labels2 = 0.5 * predictions2 + labels2.unsqueeze(0).expand(batch_size, -1)

        # in most cases, the data is constructed like predictions_1, the data of such a batch different.
        # but in a few cases, for example, predictions_2, the data of such a batch is exactly the same.
        predictions_labels_ = [(predictions1, labels1), (predictions2, labels2)]

        for predictions, labels in predictions_labels_:
            spearman_correlation.reset()
            spearman_correlation(predictions, labels)
            assert_allclose(
                spearman_formula(predictions.reshape(-1), labels.reshape(-1)),
                spearman_correlation.get_metric(),
            )

    @multi_device
    def test_masked_computation(self, device: str):
        spearman_correlation = SpearmanCorrelation()
        batch_size = 10
        num_labels = 10
        predictions1 = torch.randn(batch_size, num_labels, device=device)
        labels1 = 0.5 * predictions1 + torch.randn(batch_size, num_labels, device=device)

        predictions2 = torch.randn(1, device=device).expand(num_labels)
        predictions2 = predictions2.unsqueeze(0).expand(batch_size, -1)
        labels2 = torch.randn(1, device=device).expand(num_labels)
        labels2 = 0.5 * predictions2 + labels2.unsqueeze(0).expand(batch_size, -1)

        # in most cases, the data is constructed like predictions_1, the data of such a batch different.
        # but in a few cases, for example, predictions_2, the data of such a batch is exactly the same.
        predictions_labels_ = [(predictions1, labels1), (predictions2, labels2)]

        # Random binary mask
        mask = torch.randint(0, 2, size=(batch_size, num_labels), device=device).bool()

        for predictions, labels in predictions_labels_:
            spearman_correlation.reset()
            spearman_correlation(predictions, labels, mask)
            expected_spearman_correlation = spearman_formula(
                predictions.view(-1), labels.view(-1), mask=mask.view(-1)
            )

            # because add mask, a batch of predictions or labels will have many 0,
            # spearman correlation algorithm will dependence the sorting position of a set of numbers,
            # too many identical numbers will result in different calculation results each time
            # but the positive and negative results are the same,
            # so here we only test the positive and negative results of the results.
            assert (expected_spearman_correlation * spearman_correlation.get_metric()) > 0

    @multi_device
    def test_reset(self, device: str):
        spearman_correlation = SpearmanCorrelation()
        batch_size = 10
        num_labels = 10
        predictions = torch.randn(batch_size, num_labels, device=device)
        labels = 0.5 * predictions + torch.randn(batch_size, num_labels, device=device)

        # 1.test spearman_correlation.reset()
        spearman_correlation.reset()
        spearman_correlation(predictions, labels)
        temp = spearman_correlation.get_metric()
        spearman_correlation.reset()
        spearman_correlation(predictions, labels)
        assert spearman_correlation.get_metric() == temp

        # 2.test spearman_correlation.reset()
        spearman_correlation.reset()
        spearman_correlation(predictions, labels)

        spearman_correlation.get_metric(reset=False)
        assert spearman_correlation.get_metric() != float("NaN")
        spearman_correlation.get_metric(reset=True)
        assert math.isnan(spearman_correlation.get_metric())
