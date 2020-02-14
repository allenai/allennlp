"""
A toy example of how one might train a GAN using AllenNLP.
    def set_stage(self, stage: str) -

Based on https://github.com/devnag/pytorch-generative-adversarial-networks.

We use a dataset reader to sample from both the "true" distribution N(4, 1.25),
and from uniform noise. We'll then adversarially train a generator `Model`
to transform the noise into something that (hopefully) looks like the true distribution
and a discriminator `Model` to (hopefully) distinguish between the "true" and generated data.
"""
import tempfile
from typing import Iterable, Iterator, List, Optional

import torch

from allennlp.commands.train import train_model
from allennlp.common import Lazy, Params, Registrable
from allennlp.common.testing import ModelTestCase
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances, DatasetReader
from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.tests.training.gan_trainer_test import InputSampler
from allennlp.training import util as training_util
from allennlp.training.callback_trainer import CallbackTrainer
from allennlp.training.callbacks import Callback, Events, handle_event
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase


@Model.register("gan")
class Gan(Model):
    """
    Our trainer wants a single model, so we cheat by encapsulating both the
    generator and discriminator inside a single model. We'll access them individually.
    """

    def __init__(self, generator: Model, discriminator: Model) -> None:
        super().__init__(None)
        self.generator = generator
        self.discriminator = discriminator


@Optimizer.register("gan", constructor="from_partial_objects")
class GanOptimizer(torch.optim.Optimizer, Registrable):
    """
    Similarly, we want different optimizers for the generator and discriminator,
    so we cheat by encapsulating both in a single optimizer that has a state
    indicating which one to use.
    """

    def __init__(
        self,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
    ) -> None:
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.stage = ""
        self.param_groups = {}  # type: ignore

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
    def from_partial_objects(
        cls,
        model: Gan,
        generator_optimizer: Lazy[Optimizer],
        discriminator_optimizer: Lazy[Optimizer],
    ) -> "GanOptimizer":
        # Because we "tagged" the parameters, we can use getattr to figure out
        # which ones go with which model.
        generator_parameters = [
            [n, p] for n, p in model.generator.named_parameters() if p.requires_grad
        ]
        discriminator_parameters = [
            [n, p] for n, p in model.discriminator.named_parameters() if p.requires_grad
        ]

        generator_optimizer_ = generator_optimizer.construct(model_parameters=generator_parameters)
        discriminator_optimizer_ = discriminator_optimizer.construct(
            model_parameters=discriminator_parameters
        )

        return cls(
            generator_optimizer=generator_optimizer_,
            discriminator_optimizer=discriminator_optimizer_,
        )


GanOptimizer.register("gan", constructor="from_partial_objects")(GanOptimizer)


@DatasetReader.register("gan-callback")
class GanCallbackDatasetReader(DatasetReader):
    def __init__(
        self,
        sampler: InputSampler,
        noise_sampler: InputSampler,
        batch_size: int,
        batches_per_epoch: int,
    ) -> None:
        super().__init__(lazy=False)
        self.sampler = sampler
        self.noise_sampler = noise_sampler
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def _make_instances(self, stage: str) -> Iterator[Instance]:
        sampler = self.sampler if stage == "discriminator_real" else self.noise_sampler
        stage_field = MetadataField(stage)

        for _ in range(self.batch_size):
            array_field = ArrayField(sampler.sample(1))
            yield Instance({"array": array_field, "stage": stage_field})

    def _one_epoch(self) -> Iterator[Instance]:
        for _ in range(self.batches_per_epoch):
            yield from self._make_instances("discriminator_real")
            yield from self._make_instances("discriminator_fake")
            yield from self._make_instances("generator")

    def read(self, file_path: str) -> Iterable[Instance]:
        return _LazyInstances(self._one_epoch)


@Callback.register("track-gan-metrics")
class TrainGan(Callback):
    @handle_event(Events.BATCH_END)
    def compute_metrics(self, trainer: "GanCallbackTrainer"):

        trainer.train_metrics = {
            "dfl": trainer.discriminator_fake_loss,
            "drl": trainer.discriminator_real_loss,
            "gl": trainer.discriminator_real_loss,
            "mean": trainer.fake_mean / max(trainer.count, 1),
            "stdev": trainer.fake_stdev / max(trainer.count, 1),
        }


def config(sample_size: int = 500, batches_per_epoch: int = 40, num_epochs: int = 50) -> Params:
    return Params(
        {
            "dataset_reader": {
                "type": "gan-callback",
                "batch_size": sample_size,
                "batches_per_epoch": batches_per_epoch,
                "sampler": {"type": "normal", "mean": 4.0, "stdev": 1.25},
                "noise_sampler": {"type": "uniform"},
            },
            "iterator": {"type": "basic", "batch_size": sample_size},
            "train_data_path": "",
            "model": {
                "type": "gan",
                "generator": {
                    "type": "generator-test",
                    "input_dim": 1,
                    "hidden_dim": 5,
                    "output_dim": 1,
                },
                "discriminator": {
                    "type": "discriminator-test",
                    "input_dim": sample_size,
                    "hidden_dim": 5,
                    "preprocessing": "moments",
                },
            },
            "trainer": {
                "type": "gan-callback",
                "optimizer": {
                    "type": "gan",
                    "generator_optimizer": {"type": "sgd", "lr": 0.05},
                    "discriminator_optimizer": {"type": "sgd", "lr": 0.05},
                },
                "callbacks": [
                    "track-gan-metrics",
                    {"type": "gradient_norm_and_clip", "grad_norm": 1.0},
                ],
                "num_epochs": num_epochs,
            },
        }
    )


@TrainerBase.register("gan-callback", constructor="from_partial_objects")
class GanCallbackTrainer(CallbackTrainer):
    def __init__(
        self,
        model: Gan,
        train_data: Iterable[Instance],
        iterator: DataIterator,
        optimizer: GanOptimizer,
        num_epochs: int = 20,
        shuffle: bool = False,
        serialization_dir: Optional[str] = None,
        cuda_device: int = -1,
        callbacks: List[Callback] = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__(
            model,
            train_data,
            iterator,
            optimizer,
            num_epochs,
            shuffle,
            serialization_dir,
            cuda_device,
            callbacks,
        )
        # Need to track our own metrics as well
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.generator_loss = 0.0
        self.discriminator_real_loss = 0.0
        self.discriminator_fake_loss = 0.0
        self.fake_mean = 0.0
        self.fake_stdev = 0.0
        self.count = 0

    def train_one_batch(self, batch):

        array = batch["array"]
        # We should not have mixed batches:
        if len(set(batch["stage"])) != 1:
            raise ValueError("mixed batch")

        stage = batch["stage"][0]
        self.optimizer.stage = stage
        self.optimizer.zero_grad()

        if stage == "discriminator_real":
            # Generate real data and expect the discriminator to predict 1.
            output = self.model.discriminator(array, torch.ones(1))
            loss = output["loss"]
            self.discriminator_real_loss += loss.sum().item()
        elif stage == "discriminator_fake":
            # Generate fake data and expect the discriminator to predict 0.
            fake_data = self.model.generator(array)
            output = self.model.discriminator(fake_data["output"], torch.zeros(1))
            loss = output["loss"]
            self.discriminator_fake_loss += loss.sum().item()
        elif stage == "generator":
            # Generate fake data and try to fool the discriminator.
            generated = self.model.generator(array, self.model.discriminator)
            fake_data = generated["output"]
            loss = generated["loss"]
            self.generator_loss += loss.sum().item()

            self.fake_mean += fake_data.mean()
            self.fake_stdev += fake_data.std()
            self.count += 1

        self.train_loss += loss.sum().item()
        loss.backward()

        count = max(self.count, 1)
        self.train_metrics = {
            "gl": self.generator_loss / count,
            "dfl": self.discriminator_fake_loss / count,
            "drl": self.discriminator_real_loss / count,
            "mean": self.fake_mean / count,
            "stdev": self.fake_stdev / count,
        }

        self.optimizer.step()

        return training_util.description_from_metrics(self.train_metrics)

    def train_one_epoch(self) -> None:
        # Reset epoch counters
        self._reset_counters()

        # Will call `self.train_one_batch`
        super().train_one_epoch()

    @classmethod
    def from_partial_objects(  # type: ignore
        cls,
        model: Gan,
        train_data: Iterable[Instance],
        iterator: DataIterator,
        optimizer: Lazy[GanOptimizer],
        num_epochs: int = 20,
        shuffle: bool = False,
        serialization_dir: Optional[str] = None,
        cuda_device: int = -1,
        callbacks: List[Callback] = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> "GanCallbackTrainer":
        optimizer = optimizer.construct()
        return cls(
            model=model,
            train_data=train_data,
            iterator=iterator,
            optimizer=optimizer,
            num_epochs=num_epochs,
            shuffle=shuffle,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            callbacks=callbacks,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )


class GanCallbackTrainerTest(ModelTestCase):
    def test_gan_can_train(self):
        params = config(batches_per_epoch=2, num_epochs=2)
        train_model(params, self.TEST_DIR)


if __name__ == "__main__":
    # Run it yourself, it's fun!
    #
    # python -m allennlp.tests.training.gan_callback_trainer_test
    #

    serialization_dir = tempfile.mkdtemp()

    params = config()
    trainer = TrainerBase.from_params(params=params, serialization_dir=serialization_dir)

    metrics = trainer.train()
    print(metrics)
