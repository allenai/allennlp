"""
A toy example of how one might train a GAN using AllenNLP.

Based on https://github.com/devnag/pytorch-generative-adversarial-networks.

We use one dataset reader to sample from the "true" distribution N(4, 1.25),
and a second to sample uniform noise. We'll then adversarially train a generator `Model`
to transform the noise into something that (hopefully) looks like the true distribution
and a discriminator `Model` to (hopefully) distinguish between the "true" and generated data.
"""
from typing import Dict, Iterable, Any

import tqdm
import torch
import numpy as np

from allennlp.common import Lazy, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase


class InputSampler(Registrable):
    """
    Abstract base class for sampling from a distribution.
    """

    def sample(self, *dims: int) -> np.ndarray:
        raise NotImplementedError


@InputSampler.register("uniform")
class UniformSampler(InputSampler):
    """
    Sample from the uniform [0, 1] distribution.
    """

    def sample(self, *dims: int) -> np.ndarray:
        return np.random.uniform(0, 1, dims)


@InputSampler.register("normal")
class NormalSampler(InputSampler):
    """
    Sample from the normal distribution.
    """

    def __init__(self, mean: float = 0, stdev: float = 1.0) -> None:
        self.mean = mean
        self.stdev = stdev

    def sample(self, *dims: int) -> np.ndarray:
        return np.random.normal(self.mean, self.stdev, dims)


@DatasetReader.register("sampling")
class SamplingReader(DatasetReader):
    """
    A dataset reader that just samples from the provided sampler forever.
    """

    def __init__(self, sampler: InputSampler) -> None:
        super().__init__(lazy=True)
        self.sampler = sampler

    def _read(self, _: str) -> Iterable[Instance]:
        while True:
            example = self.sampler.sample(1)
            yield self.text_to_instance(example)

    def text_to_instance(self, example: np.ndarray) -> Instance:  # type: ignore

        field = ArrayField(example)
        return Instance({"array": field})


@Model.register("generator-test")
class Generator(Model):
    """
    A model that takes random noise (batch_size, input_dim)
    and transforms it to (batch_size, output_dim).

    If its forward pass is provided with a discriminator,
    it computes a loss based on the idea that it wants
    to trick the discriminator into predicting that its output is genuine.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: Activation = torch.nn.Tanh(),
    ) -> None:
        super().__init__(None)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = activation
        self.loss = torch.nn.BCELoss()

    def forward(  # type: ignore
        self, inputs: torch.Tensor, discriminator: Model = None
    ) -> Dict[str, torch.Tensor]:

        hidden1 = self.activation(self.linear1(inputs))
        hidden2 = self.activation(self.linear2(hidden1))
        output = self.linear3(hidden2)
        output_dict = {"output": output}

        if discriminator is not None:
            predicted = discriminator(output)["output"]
            # We desire for the discriminator to think this is real.
            desired = torch.ones_like(predicted)
            output_dict["loss"] = self.loss(predicted, desired)

        return output_dict


def get_moments(dist: torch.Tensor) -> torch.Tensor:
    """
    Returns the first 4 moments of the input data.
    We'll (potentially) use this as the input to our discriminator.
    """
    mean = torch.mean(dist)
    diffs = dist - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = (
        torch.mean(torch.pow(zscores, 4.0)) - 3.0
    )  # excess kurtosis, should be 0 for Gaussian
    final = torch.cat((mean.reshape(1), std.reshape(1), skews.reshape(1), kurtoses.reshape(1)))
    return final


@Model.register("discriminator-test")
class Discriminator(Model):
    """
    A model that takes a sample (input_dim,) and tries to predict 1
    if it's from the true distribution and 0 if it's from the generator.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: Activation = torch.nn.Sigmoid(),
        preprocessing: str = None,
    ) -> None:
        super().__init__(None)
        if preprocessing is None:
            self.preprocess = lambda x: x
        elif preprocessing == "moments":
            self.preprocess = get_moments
            input_dim = 4
        else:
            raise ConfigurationError("unknown preprocessing")

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)
        self.activation = activation
        self.loss = torch.nn.BCELoss()

    def forward(  # type: ignore
        self, inputs: torch.Tensor, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        inputs = inputs.squeeze(-1)
        hidden1 = self.activation(self.linear1(self.preprocess(inputs)))
        hidden2 = self.activation(self.linear2(hidden1))
        output = self.activation(self.linear3(hidden2))
        output_dict = {"output": output}
        if label is not None:
            output_dict["loss"] = self.loss(output, label)

        return output_dict


@TrainerBase.register("gan-test", constructor="from_partial_objects")
class GanTestTrainer(TrainerBase):
    def __init__(
        self,
        serialization_dir: str,
        data: Iterable[Instance],
        noise: Iterable[Instance],
        generator: Model,
        discriminator: Model,
        iterator: DataIterator,
        noise_iterator: DataIterator,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        batches_per_epoch: int,
        num_epochs: int,
    ) -> None:
        super().__init__(serialization_dir, -1)
        self.data = data
        self.noise = noise
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.num_epochs = num_epochs
        self.iterator = iterator
        self.noise_iterator = noise_iterator
        self.batches_per_epoch = batches_per_epoch

    def train_one_epoch(self) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()

        generator_loss = 0.0
        discriminator_real_loss = 0.0
        discriminator_fake_loss = 0.0
        fake_mean = 0.0
        fake_stdev = 0.0

        # First train the discriminator
        data_iterator = self.iterator(self.data)
        noise_iterator = self.noise_iterator(self.noise)

        for _ in range(self.batches_per_epoch):
            self.discriminator_optimizer.zero_grad()

            batch = next(data_iterator)
            noise = next(noise_iterator)

            # Real example, want discriminator to predict 1.
            real_error = self.discriminator(batch["array"], torch.ones(1))["loss"]
            real_error.backward()

            # Fake example, want discriminator to predict 0.
            fake_data = self.generator(noise["array"])["output"]
            fake_error = self.discriminator(fake_data, torch.zeros(1))["loss"]
            fake_error.backward()

            discriminator_real_loss += real_error.sum().item()
            discriminator_fake_loss += fake_error.sum().item()

            self.discriminator_optimizer.step()

        # Now train the generator
        for _ in range(self.batches_per_epoch):
            self.generator_optimizer.zero_grad()

            noise = next(noise_iterator)
            generated = self.generator(noise["array"], self.discriminator)
            fake_data = generated["output"]
            fake_error = generated["loss"]
            fake_error.backward()

            fake_mean += fake_data.mean()
            fake_stdev += fake_data.std()

            generator_loss += fake_error.sum().item()

            self.generator_optimizer.step()

        return {
            "generator_loss": generator_loss,
            "discriminator_fake_loss": discriminator_fake_loss,
            "discriminator_real_loss": discriminator_real_loss,
            "mean": fake_mean / self.batches_per_epoch,
            "stdev": fake_stdev / self.batches_per_epoch,
        }

    def train(self) -> Dict[str, Any]:
        with tqdm.trange(self.num_epochs) as epochs:
            for _ in epochs:
                metrics = self.train_one_epoch()
                description = (
                    f'gl: {metrics["generator_loss"]:.3f} '
                    f'dfl: {metrics["discriminator_fake_loss"]:.3f} '
                    f'drl: {metrics["discriminator_real_loss"]:.3f} '
                    f'mean: {metrics["mean"]:.2f} '
                    f'std: {metrics["stdev"]:.2f} '
                )
                epochs.set_description(description)
        return metrics

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        data_reader: DatasetReader,
        noise_reader: DatasetReader,
        generator: Model,
        discriminator: Model,
        iterator: DataIterator,
        noise_iterator: DataIterator,
        generator_optimizer: Lazy[Optimizer],
        discriminator_optimizer: Lazy[Optimizer],
        num_epochs: int,
        batches_per_epoch: int,
    ) -> "GanTestTrainer":
        data = data_reader.read("")
        noise = noise_reader.read("")

        generator_params = [[n, p] for n, p in generator.named_parameters() if p.requires_grad]
        generator_optimizer_ = generator_optimizer.construct(model_parameters=generator_params)

        discriminator_params = [
            [n, p] for n, p in discriminator.named_parameters() if p.requires_grad
        ]
        discriminator_optimizer_ = discriminator_optimizer.construct(
            model_parameters=discriminator_params
        )

        return cls(
            serialization_dir,
            data,
            noise,
            generator,
            discriminator,
            iterator,
            noise_iterator,
            generator_optimizer_,
            discriminator_optimizer_,
            batches_per_epoch,
            num_epochs,
        )


class GanTrainerTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        params = Params(
            {
                "type": "gan-test",
                "data_reader": {
                    "type": "sampling",
                    "sampler": {"type": "normal", "mean": 4.0, "stdev": 1.25},
                },
                "noise_reader": {"type": "sampling", "sampler": {"type": "uniform"}},
                "generator": {
                    "type": "generator-test",
                    "input_dim": 1,
                    "hidden_dim": 5,
                    "output_dim": 1,
                },
                "discriminator": {"type": "discriminator-test", "input_dim": 500, "hidden_dim": 10},
                "iterator": {"type": "basic", "batch_size": 500},
                "noise_iterator": {"type": "basic", "batch_size": 500},
                "generator_optimizer": {"type": "sgd", "lr": 0.1},
                "discriminator_optimizer": {"type": "sgd", "lr": 0.1},
                "num_epochs": 5,
                "batches_per_epoch": 2,
            }
        )

        self.trainer = TrainerBase.from_params(params=params, serialization_dir=self.TEST_DIR)

    def test_gan_can_train(self):
        self.trainer.train()


if __name__ == "__main__":
    # Run it yourself, it's fun!
    #
    # python -m allennlp.tests.training.gan_trainer_test
    #

    sample_size = 500

    params_ = Params(
        {
            "type": "gan-test",
            "data_reader": {
                "type": "sampling",
                "sampler": {"type": "normal", "mean": 4.0, "stdev": 1.25},
            },
            "noise_reader": {"type": "sampling", "sampler": {"type": "uniform"}},
            "generator": {
                "type": "generator-test",
                "input_dim": 1,
                "hidden_dim": 5,
                "output_dim": 1,
            },
            "discriminator": {
                "type": "discriminator-test",
                "input_dim": sample_size,
                "hidden_dim": 10,
                "preprocessing": "moments",
            },
            "iterator": {"type": "basic", "batch_size": sample_size},
            "noise_iterator": {"type": "basic", "batch_size": sample_size},
            "generator_optimizer": {"type": "sgd", "lr": 0.1},
            "discriminator_optimizer": {"type": "sgd", "lr": 0.1},
            "num_epochs": 1000,
            "batches_per_epoch": 2,
        }
    )

    import tempfile

    serialization_dir_ = tempfile.mkdtemp()
    trainer_ = TrainerBase.from_params(params=params_, serialization_dir=serialization_dir_)
    metrics_ = trainer_.train()
    print(metrics_)
