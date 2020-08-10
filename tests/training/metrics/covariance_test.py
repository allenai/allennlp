import numpy as np
import torch
from torch.testing import assert_allclose

from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    run_distributed_test,
)
from allennlp.training.metrics import Covariance


class CovarianceTest(AllenNlpTestCase):
    @multi_device
    def test_covariance_unmasked_computation(self, device: str):
        covariance = Covariance()
        batch_size = 100
        num_labels = 10
        predictions = torch.randn(batch_size, num_labels, device=device)
        labels = 0.5 * predictions + torch.randn(batch_size, num_labels, device=device)

        stride = 10

        for i in range(batch_size // stride):
            timestep_predictions = predictions[stride * i : stride * (i + 1), :]
            timestep_labels = labels[stride * i : stride * (i + 1), :]
            # Flatten the predictions and labels thus far, so numpy treats them as
            # independent observations.
            expected_covariance = np.cov(
                predictions[: stride * (i + 1), :].view(-1).cpu().numpy(),
                labels[: stride * (i + 1), :].view(-1).cpu().numpy(),
            )[0, 1]
            covariance(timestep_predictions, timestep_labels)
            assert_allclose(expected_covariance, covariance.get_metric())

        # Test reset
        covariance.reset()
        covariance(predictions, labels)
        assert_allclose(
            np.cov(predictions.view(-1).cpu().numpy(), labels.view(-1).cpu().numpy())[0, 1],
            covariance.get_metric(),
        )

    @multi_device
    def test_covariance_masked_computation(self, device: str):
        covariance = Covariance()
        batch_size = 100
        num_labels = 10
        predictions = torch.randn(batch_size, num_labels, device=device)
        labels = 0.5 * predictions + torch.randn(batch_size, num_labels, device=device)
        # Random binary mask
        mask = torch.randint(0, 2, size=(batch_size, num_labels), device=device).bool()
        stride = 10

        for i in range(batch_size // stride):
            timestep_predictions = predictions[stride * i : stride * (i + 1), :]
            timestep_labels = labels[stride * i : stride * (i + 1), :]
            timestep_mask = mask[stride * i : stride * (i + 1), :]
            # Flatten the predictions, labels, and mask thus far, so numpy treats them as
            # independent observations.
            expected_covariance = np.cov(
                predictions[: stride * (i + 1), :].view(-1).cpu().numpy(),
                labels[: stride * (i + 1), :].view(-1).cpu().numpy(),
                fweights=mask[: stride * (i + 1), :].view(-1).cpu().numpy(),
            )[0, 1]
            covariance(timestep_predictions, timestep_labels, timestep_mask)
            assert_allclose(expected_covariance, covariance.get_metric())

        # Test reset
        covariance.reset()
        covariance(predictions, labels, mask)
        assert_allclose(
            np.cov(
                predictions.view(-1).cpu().numpy(),
                labels.view(-1).cpu().numpy(),
                fweights=mask.view(-1).cpu().numpy(),
            )[0, 1],
            covariance.get_metric(),
        )

    def test_distributed_covariance(self):
        batch_size = 10
        num_labels = 10
        predictions = torch.randn(batch_size, num_labels)
        labels = 0.5 * predictions + torch.randn(batch_size, num_labels)
        # Random binary mask
        mask = torch.randint(0, 2, size=(batch_size, num_labels)).bool()

        expected_covariance = np.cov(
            predictions.view(-1).cpu().numpy(),
            labels.view(-1).cpu().numpy(),
            fweights=mask.view(-1).cpu().numpy(),
        )[0, 1]

        predictions = [predictions[:5], predictions[5:]]
        labels = [labels[:5], labels[5:]]
        mask = [mask[:5], mask[5:]]

        metric_kwargs = {"predictions": predictions, "gold_labels": labels, "mask": mask}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            Covariance(),
            metric_kwargs,
            expected_covariance,
            exact=(0.0001, 1e-01),
        )

    def test_distributed_covariance_unequal_batches(self):
        batch_size = 10
        num_labels = 10
        predictions = torch.randn(batch_size, num_labels)
        labels = 0.5 * predictions + torch.randn(batch_size, num_labels)
        # Random binary mask
        mask = torch.randint(0, 2, size=(batch_size, num_labels)).bool()

        expected_covariance = np.cov(
            predictions.view(-1).cpu().numpy(),
            labels.view(-1).cpu().numpy(),
            fweights=mask.view(-1).cpu().numpy(),
        )[0, 1]

        predictions = [predictions[:6], predictions[6:]]
        labels = [labels[:6], labels[6:]]
        mask = [mask[:6], mask[6:]]

        metric_kwargs = {"predictions": predictions, "gold_labels": labels, "mask": mask}
        run_distributed_test(
            [-1, -1],
            global_distributed_metric,
            Covariance(),
            metric_kwargs,
            expected_covariance,
            exact=(0.0001, 1e-01),
        )
