# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import numpy as np
from numpy.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import PearsonCorrelation


def pearson_corrcoef(predictions, labels, fweights=None):
    covariance_matrices = np.cov(predictions, labels, fweights=fweights)
    denominator = np.sqrt(covariance_matrices[0, 0] * covariance_matrices[1, 1])
    if np.around(denominator, decimals=5) == 0:
        expected_pearson_correlation = 0
    else:
        expected_pearson_correlation = covariance_matrices[0, 1] / denominator
    return expected_pearson_correlation


class PearsonCorrelationTest(AllenNlpTestCase):
    def test_pearson_correlation_unmasked_computation(self):
        pearson_correlation = PearsonCorrelation()
        batch_size = 100
        num_labels = 10
        predictions_1 = np.random.randn(batch_size, num_labels).astype("float32")
        labels_1 = 0.5 * predictions_1 + np.random.randn(batch_size, num_labels).astype("float32")

        predictions_2 = np.random.randn(1).repeat(num_labels).astype("float32")
        predictions_2 = predictions_2[np.newaxis, :].repeat(batch_size, axis=0)
        labels_2 = np.random.randn(1).repeat(num_labels).astype("float32")
        labels_2 = 0.5 * predictions_2 + labels_2[np.newaxis, :].repeat(batch_size, axis=0)

        # in most cases, the data is constructed like predictions_1, the data of such a batch different.
        # but in a few cases, for example, predictions_2, the data of such a batch is exactly the same.
        predictions_labels = [(predictions_1, labels_1), (predictions_2, labels_2)]

        stride = 10

        for predictions, labels in predictions_labels:
            pearson_correlation.reset()
            for i in range(batch_size // stride):
                timestep_predictions = torch.FloatTensor(predictions[stride * i:stride * (i + 1), :])
                timestep_labels = torch.FloatTensor(labels[stride * i:stride * (i + 1), :])
                expected_pearson_correlation = pearson_corrcoef(predictions[:stride * (i + 1), :].reshape(-1),
                                                                labels[:stride * (i + 1), :].reshape(-1))
                pearson_correlation(timestep_predictions, timestep_labels)
                assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric(), rtol=1e-5)
            # Test reset
            pearson_correlation.reset()
            pearson_correlation(torch.FloatTensor(predictions), torch.FloatTensor(labels))
            assert_allclose(pearson_corrcoef(predictions.reshape(-1), labels.reshape(-1)),
                            pearson_correlation.get_metric(), rtol=1e-5)

    def test_pearson_correlation_masked_computation(self):
        pearson_correlation = PearsonCorrelation()
        batch_size = 100
        num_labels = 10
        predictions_1 = np.random.randn(batch_size, num_labels).astype("float32")
        labels_1 = 0.5 * predictions_1 + np.random.randn(batch_size, num_labels).astype("float32")

        predictions_2 = np.random.randn(1).repeat(num_labels).astype("float32")
        predictions_2 = predictions_2[np.newaxis, :].repeat(batch_size, axis=0)
        labels_2 = np.random.randn(1).repeat(num_labels).astype("float32")
        labels_2 = 0.5 * predictions_2 + labels_2[np.newaxis, :].repeat(batch_size, axis=0)

        predictions_labels = [(predictions_1, labels_1), (predictions_2, labels_2)]

        # Random binary mask
        mask = np.random.randint(0, 2, size=(batch_size, num_labels)).astype("float32")
        stride = 10

        for predictions, labels in predictions_labels:
            pearson_correlation.reset()
            for i in range(batch_size // stride):
                timestep_predictions = torch.FloatTensor(predictions[stride * i:stride * (i + 1), :])
                timestep_labels = torch.FloatTensor(labels[stride * i:stride * (i + 1), :])
                timestep_mask = torch.FloatTensor(mask[stride * i:stride * (i + 1), :])
                expected_pearson_correlation = pearson_corrcoef(predictions[:stride * (i + 1), :].reshape(-1),
                                                                labels[:stride * (i + 1), :].reshape(-1),
                                                                fweights=mask[:stride * (i + 1), :].reshape(-1))

                pearson_correlation(timestep_predictions, timestep_labels, timestep_mask)
                assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric(), rtol=1e-5)
            # Test reset
            pearson_correlation.reset()
            pearson_correlation(torch.FloatTensor(predictions),
                                torch.FloatTensor(labels), torch.FloatTensor(mask))
            expected_pearson_correlation = pearson_corrcoef(predictions.reshape(-1), labels.reshape(-1),
                                                            fweights=mask.reshape(-1))

            assert_allclose(expected_pearson_correlation, pearson_correlation.get_metric(), rtol=1e-5)
