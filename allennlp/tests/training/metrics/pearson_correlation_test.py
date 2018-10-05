# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import numpy as np
from numpy.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import PearsonCorrelation


class PearsonCorrelationTest(AllenNlpTestCase):
    def test_pearson_correlation_unmasked_computation(self):
        pearson_correlation = PearsonCorrelation()
        batch_size = 100
        num_labels = 10
        predictions = np.random.randn(batch_size, num_labels).astype("float32")
        labels = 0.5 * predictions + np.random.randn(batch_size, num_labels).astype("float32")

        stride = 10

        for i in range(batch_size // stride):
            timestep_predictions = torch.FloatTensor(predictions[stride * i:stride * (i+1), :])
            timestep_labels = torch.FloatTensor(labels[stride * i:stride * (i+1), :])
            expected_pearson_correlation = np.corrcoef(predictions[:stride * (i + 1), :].reshape(-1),
                                                       labels[:stride * (i + 1), :].reshape(-1))[0, 1]
            pearson_correlation(timestep_predictions, timestep_labels)
            assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric(), rtol=1e-5)
        # Test reset
        pearson_correlation.reset()
        pearson_correlation(torch.FloatTensor(predictions), torch.FloatTensor(labels))
        assert_allclose(np.corrcoef(predictions.reshape(-1), labels.reshape(-1))[0, 1],
                        pearson_correlation.get_metric(), rtol=1e-5)

    def test_pearson_correlation_masked_computation(self):
        pearson_correlation = PearsonCorrelation()
        batch_size = 100
        num_labels = 10
        predictions = np.random.randn(batch_size, num_labels).astype("float32")
        labels = 0.5 * predictions + np.random.randn(batch_size, num_labels).astype("float32")
        # Random binary mask
        mask = np.random.randint(0, 2, size=(batch_size, num_labels)).astype("float32")
        stride = 10

        for i in range(batch_size // stride):
            timestep_predictions = torch.FloatTensor(predictions[stride * i:stride * (i+1), :])
            timestep_labels = torch.FloatTensor(labels[stride * i:stride * (i+1), :])
            timestep_mask = torch.FloatTensor(mask[stride * i:stride * (i+1), :])
            covariance_matrices = np.cov(predictions[:stride * (i + 1), :].reshape(-1),
                                         labels[:stride * (i + 1), :].reshape(-1),
                                         fweights=mask[:stride * (i + 1), :].reshape(-1))
            expected_pearson_correlation = covariance_matrices[0, 1] / np.sqrt(covariance_matrices[0, 0] *
                                                                               covariance_matrices[1, 1])
            pearson_correlation(timestep_predictions, timestep_labels, timestep_mask)
            assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric(), rtol=1e-5)
        # Test reset
        pearson_correlation.reset()
        pearson_correlation(torch.FloatTensor(predictions), torch.FloatTensor(labels), torch.FloatTensor(mask))
        covariance_matrices = np.cov(predictions.reshape(-1), labels.reshape(-1), fweights=mask.reshape(-1))
        expected_pearson_correlation = covariance_matrices[0, 1] / np.sqrt(covariance_matrices[0, 0] *
                                                                           covariance_matrices[1, 1])
        assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric(), rtol=1e-5)
