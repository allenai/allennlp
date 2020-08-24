from typing import Optional

import numpy as np
import torch
from torch.testing import assert_allclose

from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    # global_distributed_metric,
    # run_distributed_test,
)
from allennlp.training.metrics import PearsonCorrelation


def pearson_corrcoef(
    predictions: np.ndarray, labels: np.ndarray, fweights: Optional[np.ndarray] = None
):
    covariance_matrices = np.cov(predictions, labels, fweights=fweights)
    denominator = np.sqrt(covariance_matrices[0, 0] * covariance_matrices[1, 1])
    if np.around(denominator, decimals=5) == 0:
        expected_pearson_correlation = 0
    else:
        expected_pearson_correlation = covariance_matrices[0, 1] / denominator
    return expected_pearson_correlation


class PearsonCorrelationTest(AllenNlpTestCase):
    @multi_device
    def test_pearson_correlation_unmasked_computation(self, device: str):
        pearson_correlation = PearsonCorrelation()
        batch_size = 100
        num_labels = 10
        predictions_1 = torch.randn(batch_size, num_labels, device=device)
        labels_1 = 0.5 * predictions_1 + torch.randn(batch_size, num_labels, device=device)

        predictions_2 = torch.randn(1, device=device).expand(num_labels)
        predictions_2 = predictions_2.unsqueeze(0).expand(batch_size, -1)
        labels_2 = torch.randn(1, device=device).expand(num_labels)
        labels_2 = 0.5 * predictions_2 + labels_2.unsqueeze(0).expand(batch_size, -1)

        # in most cases, the data is constructed like predictions_1, the data of such a batch different.
        # but in a few cases, for example, predictions_2, the data of such a batch is exactly the same.
        predictions_labels = [(predictions_1, labels_1), (predictions_2, labels_2)]

        stride = 10

        for predictions, labels in predictions_labels:
            pearson_correlation.reset()
            for i in range(batch_size // stride):
                timestep_predictions = predictions[stride * i : stride * (i + 1), :]
                timestep_labels = labels[stride * i : stride * (i + 1), :]
                expected_pearson_correlation = pearson_corrcoef(
                    predictions[: stride * (i + 1), :].view(-1).cpu().numpy(),
                    labels[: stride * (i + 1), :].view(-1).cpu().numpy(),
                )
                pearson_correlation(timestep_predictions, timestep_labels)
                assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric())
            # Test reset
            pearson_correlation.reset()
            pearson_correlation(predictions, labels)
            assert_allclose(
                pearson_corrcoef(predictions.view(-1).cpu().numpy(), labels.view(-1).cpu().numpy()),
                pearson_correlation.get_metric(),
            )

    @multi_device
    def test_pearson_correlation_masked_computation(self, device: str):
        pearson_correlation = PearsonCorrelation()
        batch_size = 100
        num_labels = 10
        predictions_1 = torch.randn(batch_size, num_labels, device=device)
        labels_1 = 0.5 * predictions_1 + torch.randn(batch_size, num_labels, device=device)

        predictions_2 = torch.randn(1, device=device).expand(num_labels)
        predictions_2 = predictions_2.unsqueeze(0).expand(batch_size, -1)
        labels_2 = torch.randn(1, device=device).expand(num_labels)
        labels_2 = 0.5 * predictions_2 + labels_2.unsqueeze(0).expand(batch_size, -1)

        predictions_labels = [(predictions_1, labels_1), (predictions_2, labels_2)]

        # Random binary mask
        mask = torch.randint(0, 2, size=(batch_size, num_labels), device=device).bool()
        stride = 10

        for predictions, labels in predictions_labels:
            pearson_correlation.reset()
            for i in range(batch_size // stride):
                timestep_predictions = predictions[stride * i : stride * (i + 1), :]
                timestep_labels = labels[stride * i : stride * (i + 1), :]
                timestep_mask = mask[stride * i : stride * (i + 1), :]
                expected_pearson_correlation = pearson_corrcoef(
                    predictions[: stride * (i + 1), :].view(-1).cpu().numpy(),
                    labels[: stride * (i + 1), :].view(-1).cpu().numpy(),
                    fweights=mask[: stride * (i + 1), :].view(-1).cpu().numpy(),
                )

                pearson_correlation(timestep_predictions, timestep_labels, timestep_mask)
                assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric())
            # Test reset
            pearson_correlation.reset()
            pearson_correlation(predictions, labels, mask)
            expected_pearson_correlation = pearson_corrcoef(
                predictions.view(-1).cpu().numpy(),
                labels.view(-1).cpu().numpy(),
                fweights=mask.view(-1).cpu().numpy(),
            )

            assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric())

    # Commenting in order to revisit distributed covariance (on which PearsonCorrelation depends) later.

    # def test_distributed_pearson(self):
    #     batch_size = 10
    #     num_labels = 10
    #     predictions = torch.randn(batch_size, num_labels)
    #     labels = 0.5 * predictions + torch.randn(batch_size, num_labels)

    #     expected_pearson_correlation = pearson_corrcoef(
    #         predictions.view(-1).cpu().numpy(), labels.view(-1).cpu().numpy(),
    #     )
    #     predictions = [predictions[:5], predictions[5:]]
    #     labels = [labels[:5], labels[5:]]
    #     metric_kwargs = {"predictions": predictions, "gold_labels": labels}
    #     run_distributed_test(
    #         [-1, -1],
    #         global_distributed_metric,
    #         PearsonCorrelation(),
    #         metric_kwargs,
    #         expected_pearson_correlation,
    #         exact=(0.0001, 1e-01),
    #     )
