"""
A toy example of how one might train a GAN using AllenNLP.
    def set_stage(self, stage: str) -

Based on https://github.com/devnag/pytorch-generative-adversarial-networks.

We use one dataset reader to sample from the "true" distribution N(4, 1.25),
and a second to sample uniform noise. We'll then adversarially train a generator `Model`
to transform the noise into something that (hopefully) looks like the true distribution
and a discriminator `Model` to (hopefully) distinguish between the "true" and generated data.
"""
# pylint: disable=bad-continuation

from typing import Dict, Iterable, List
import tempfile

import torch
import numpy as np

from _pytest.monkeypatch import MonkeyPatch

from allennlp.commands.train import train_model
from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.training.callbacks import Callback, Events, handle_event
from allennlp.training.optimizers import Optimizer


class InputSampler(Registrable):
    """
    Abstract base class for sampling from a distribution.
    """
    def sample(self, *dims: int) -> np.ndarray:
        raise NotImplementedError


@InputSampler.register('uniform')
class UniformSampler(InputSampler):
    """
    Sample from the uniform [0, 1] distribution.
    """
    def sample(self, *dims: int) -> np.ndarray:
        return np.random.uniform(0, 1, dims)


@InputSampler.register('normal')
class NormalSampler(InputSampler):
    """
    Sample from the normal distribution.
    """
    def __init__(self, mean: float = 0, stdev: float = 1.0) -> None:
        self.mean = mean
        self.stdev = stdev

    def sample(self, *dims: int) -> np.ndarray:
        return np.random.normal(self.mean, self.stdev, dims)


@Model.register("generator2-test")
class Generator(Model):
    """
    A model that takes random noise (batch_size, input_dim)
    and transforms it to (batch_size, output_dim).

    If its forward pass is provided with a discriminator,
    it computes a loss based on the idea that it wants
    to trick the discriminator into predicting that its output is genuine.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 activation: Activation = torch.nn.Tanh()) -> None:
        super().__init__(None)
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = activation
        self.loss = torch.nn.BCELoss()

    def forward(self,  # type: ignore
                inputs: torch.Tensor, discriminator: Model = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
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
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
    final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
    return final


@Model.register("discriminator2-test")
class Discriminator(Model):
    """
    A model that takes a sample (input_dim,) and tries to predict 1
    if it's from the true distribution and 0 if it's from the generator.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 activation: Activation = torch.nn.Sigmoid(),
                 preprocessing: str = None) -> None:
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

    def forward(self, # type: ignore
                inputs: torch.Tensor, label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        inputs = inputs.squeeze(-1)
        hidden1 = self.activation(self.linear1(self.preprocess(inputs)))
        hidden2 = self.activation(self.linear2(hidden1))
        output = self.activation(self.linear3(hidden2))
        output_dict = {"output": output}
        if label is not None:
            output_dict["loss"] = self.loss(output, label)

        return output_dict


@Model.register("gan")
class Gan(Model):
    """
    Our trainer wants a single model, so we cheat by encapsulating both the
    generator and discriminator inside a single model. We'll access them individually.
    """
    # pylint: disable=abstract-method
    def __init__(self,
                 generator: Model,
                 discriminator: Model) -> None:
        super().__init__(None)
        self.generator = generator
        self.discriminator = discriminator

def make_batch(sampler: InputSampler, batch_size: int, stage: str) -> Batch:
    return Batch([
        Instance({
            "array": ArrayField(sampler.sample(1)),
            "stage": MetadataField(stage)
        })
        for _ in range(batch_size)
    ])

@Optimizer.register("gan")
class GanOptimizer(torch.optim.Optimizer):
    """
    Similarly, we want different optimizers for the generator and discriminator,
    so we cheat by encapsulating both in a single optimizer that has a state
    indicating which one to use.
    """
    # pylint: disable=super-init-not-called,arguments-differ
    def __init__(self,
                 generator_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.Optimizer) -> None:
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.stage = ""

    def step(self, _closure=None) -> None:
        if "discriminator" in self.stage:
            self.discriminator_optimizer.step(_closure)
        elif "generator" in self.stage:
            self.generator_optimizer.step(_closure)
        else:
            raise ValueError("unknown stage")

    def zero_grad(self) -> None:
        """
        Just zero out all the gradients.
        """
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    @classmethod
    def from_params(cls, model_parameters: List, params: Params) -> 'GanOptimizer':
        # This is a slight abuse, because we want different optimizers to have different params.
        generator_parameters = [[n, p] for n, p in model_parameters if n.startswith("generator.")]
        discriminator_parameters = [[n, p] for n, p in model_parameters if n.startswith("discriminator.")]

        generator_optimizer = Optimizer.from_params(generator_parameters, params.pop("generator_optimizer"))
        discriminator_optimizer = Optimizer.from_params(discriminator_parameters,
                                                        params.pop("discriminator_optimizer"))

        return cls(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)


@Callback.register("generate-gan-training-batches")
class GenerateGanTrainingBatches(Callback):
    def __init__(self,
                 sampler: InputSampler,
                 noise_sampler: InputSampler,
                 batch_size: int,
                 batches_per_epoch: int) -> None:
        self.sampler = sampler
        self.noise_sampler = noise_sampler
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def batches(self) -> Iterable[Batch]:
        for _ in range(self.batches_per_epoch):
            yield make_batch(self.sampler, self.batch_size, "discriminator_real")
            yield make_batch(self.noise_sampler, self.batch_size, "discriminator_fake")
            yield make_batch(self.noise_sampler, self.batch_size, "generator")

    @handle_event(Events.EPOCH_START)
    def setup_batches(self, trainer):
        trainer.training_batches = ([batch.as_tensor_dict()] for batch in self.batches())
        trainer.num_training_batches = self.batches_per_epoch * 3


@Callback.register("train-gan")
class TrainGan(Callback):
    def __init__(self) -> None:
        self.generator_loss = 0.0
        self.discriminator_real_loss = 0.0
        self.discriminator_fake_loss = 0.0
        self.fake_mean = 0.0
        self.fake_stdev = 0.0
        self.count = 0
        self.loss = None

    @handle_event(Events.EPOCH_START)
    def reset_loss(self, _trainer):
        self.generator_loss = 0.0
        self.discriminator_real_loss = 0.0
        self.discriminator_fake_loss = 0.0
        self.fake_mean = 0.0
        self.fake_stdev = 0.0
        self.count = 0

    @handle_event(Events.BATCH_START)
    def zero_grad(self, trainer):
        # pylint: disable=no-self-use
        trainer.optimizer.zero_grad()

    @handle_event(Events.FORWARD)
    def compute_loss(self, trainer):
        batch, = trainer.batch_group
        array = batch["array"]
        stage = batch["stage"][0]
        trainer.optimizer.stage = stage

        if stage == "discriminator_real":
            # Want discriminator to predict 1
            output = trainer.model.discriminator(array, torch.ones(1))
            self.loss = output["loss"]
            self.discriminator_real_loss += self.loss.sum().item()
        elif stage == "discriminator_fake":
            # Want discriminator to predict 0
            fake_data = trainer.model.generator(array)
            output = trainer.model.discriminator(fake_data["output"], torch.zeros(1))
            self.loss = output["loss"]
            self.discriminator_fake_loss += self.loss.sum().item()
        elif stage == "generator":
            generated = trainer.model.generator(array, trainer.model.discriminator)
            fake_data = generated["output"]
            self.loss = generated["loss"]
            self.generator_loss += self.loss.sum().item()

            self.fake_mean += fake_data.mean()
            self.fake_stdev += fake_data.std()
            self.count += 1

    @handle_event(Events.BACKWARD)
    def backpropagate_errors(self, trainer):
        self.loss.backward()
        trainer.train_loss += self.loss.item()

    @handle_event(Events.AFTER_BACKWARD)
    def optimize(self, trainer):
        # pylint: disable=no-self-use
        trainer.optimizer.step()

    @handle_event(Events.BATCH_END)
    def compute_metrics(self, trainer):
        trainer.train_metrics = {
            "dfl": self.discriminator_fake_loss,
            "drl": self.discriminator_real_loss,
            "gl": self.discriminator_real_loss,
            "mean": self.fake_mean / max(self.count, 1),
            "stdev": self.fake_stdev / max(self.count, 1)
        }

# This is bad, I have to monkeypatch Optimizer.from_params
_original_optimizer_from_params = Optimizer.from_params  # pylint: disable=invalid-name

def _optimizer_from_params(model_parameters: List, params: Params) -> Optimizer:
    typ3 = params.get("type")
    if typ3 == "gan":
        generator_parameters = [[n, p] for n, p in model_parameters if n.startswith("generator.")]
        discriminator_parameters = [[n, p] for n, p in model_parameters if n.startswith("discriminator.")]

        return GanOptimizer(
                generator_optimizer=_original_optimizer_from_params(generator_parameters,
                                                                    params.pop("generator_optimizer")),
                discriminator_optimizer=_original_optimizer_from_params(discriminator_parameters,
                                                                        params.pop("discriminator_optimizer"))
        )
    else:
        return _original_optimizer_from_params(model_parameters, params)

# This is also bad, I have to provide a do-nothing dataset reader.
@DatasetReader.register("nil")
class NilDatasetReader(DatasetReader):
    # pylint: disable=abstract-method
    def _read(self, file_path):
        return [Instance({})]


def config(sample_size: int = 500,
           batches_per_epoch: int = 40,
           num_epochs: int = 50,
           learning_rate: float = 0.05) -> Params:
    return Params({
        # These are unnecessary but the TrainerPieces.from_params expects them
        "iterator": {"type": "basic"},
        "dataset_reader": {"type": "nil"},
        "train_data_path": "",

        # These are necessary
        "model": {
            "type": "gan",
            "generator": {
                "type": "generator2-test",
                "input_dim": 1,
                "hidden_dim": 5,
                "output_dim": 1
            },
            "discriminator": {
                "type": "discriminator2-test",
                "input_dim": sample_size,
                "hidden_dim": 5,
                "preprocessing": "moments"
            }
        },
        "trainer": {
            "type": "callback",
            "optimizer": {
                "type": "gan",
                "generator_optimizer": {
                    "type": "sgd",
                    "lr": learning_rate
                },
                "discriminator_optimizer": {
                    "type": "sgd",
                    "lr": learning_rate
                }
            },
            "num_epochs": num_epochs,
            "callbacks": [
                {
                    "type": "train-gan"
                },
                {
                    "type": "generate-gan-training-batches",
                    "batch_size": sample_size,
                    "batches_per_epoch": batches_per_epoch,
                    "sampler": {
                        "type": "normal",
                        "mean": 4.0,
                        "stdev": 1.25
                    },
                    "noise_sampler": {
                        "type": "uniform"
                    }
                }
            ]

        }
    })


class GanCallbackTrainerTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        MonkeyPatch().setattr(Optimizer, 'from_params', _optimizer_from_params)

    def test_gan_can_train(self):
        params = config(batches_per_epoch=2, num_epochs=2)
        train_model(params, self.TEST_DIR)


if __name__ == "__main__":
    # Run it yourself, it's fun!
    #
    # python -m allennlp.tests.training.gan_callback_trainer_test
    #
    # pylint: disable=invalid-name
    MonkeyPatch().setattr(Optimizer, 'from_params', _optimizer_from_params)


    from allennlp.training.trainer_base import TrainerBase
    serialization_dir_ = tempfile.mkdtemp()

    params_ = config()
    trainer_ = TrainerBase.from_params(params_, serialization_dir_)

    metrics_ = trainer_.train()
    print(metrics_)
