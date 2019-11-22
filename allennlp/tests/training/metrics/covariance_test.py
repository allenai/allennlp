import torch
import numpy as np
from numpy.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import Covariance


class CovarianceTest(AllenNlpTestCase):
    def test_covariance_unmasked_computation(self):
        covariance = Covariance()
        batch_size = 100
        num_labels = 10
        predictions = np.random.randn(batch_size, num_labels).astype("float32")
        labels = 0.5 * predictions + np.random.randn(batch_size, num_labels).astype("float32")

        stride = 10

        for i in range(batch_size // stride):
            timestep_predictions = torch.FloatTensor(predictions[stride * i : stride * (i + 1), :])
            timestep_labels = torch.FloatTensor(labels[stride * i : stride * (i + 1), :])
            # Flatten the predictions and labels thus far, so numpy treats them as
            # independent observations.
            expected_covariance = np.cov(
                predictions[: stride * (i + 1), :].reshape(-1),
                labels[: stride * (i + 1), :].reshape(-1),
            )[0, 1]
            covariance(timestep_predictions, timestep_labels)
            assert_allclose(expected_covariance, covariance.get_metric(), rtol=1e-5)

        # Test reset
        covariance.reset()
        covariance(torch.FloatTensor(predictions), torch.FloatTensor(labels))
        assert_allclose(
            np.cov(predictions.reshape(-1), labels.reshape(-1))[0, 1],
            covariance.get_metric(),
            rtol=1e-5,
        )

    def test_covariance_masked_computation(self):
        covariance = Covariance()
        batch_size = 100
        num_labels = 10
        predictions = np.random.randn(batch_size, num_labels).astype("float32")
        labels = 0.5 * predictions + np.random.randn(batch_size, num_labels).astype("float32")
        # Random binary mask
        mask = np.random.randint(0, 2, size=(batch_size, num_labels)).astype("float32")
        stride = 10

        for i in range(batch_size // stride):
            timestep_predictions = torch.FloatTensor(predictions[stride * i : stride * (i + 1), :])
            timestep_labels = torch.FloatTensor(labels[stride * i : stride * (i + 1), :])
            timestep_mask = torch.FloatTensor(mask[stride * i : stride * (i + 1), :])
            # Flatten the predictions, labels, and mask thus far, so numpy treats them as
            # independent observations.
            expected_covariance = np.cov(
                predictions[: stride * (i + 1), :].reshape(-1),
                labels[: stride * (i + 1), :].reshape(-1),
                fweights=mask[: stride * (i + 1), :].reshape(-1),
            )[0, 1]
            covariance(timestep_predictions, timestep_labels, timestep_mask)
            assert_allclose(expected_covariance, covariance.get_metric(), rtol=1e-5)

        # Test reset
        covariance.reset()
        covariance(
            torch.FloatTensor(predictions), torch.FloatTensor(labels), torch.FloatTensor(mask)
        )
        assert_allclose(
            np.cov(predictions.reshape(-1), labels.reshape(-1), fweights=mask.reshape(-1))[0, 1],
            covariance.get_metric(),
            rtol=1e-5,
        )
