# pylint: disable=invalid-name,too-many-public-methods,protected-access,no-member
import copy
import glob
import json
import os
import re
import time
from typing import Dict, Iterable

import torch
import responses
import pytest
import numpy as np

from allennlp.common.checks import ConfigurationError

from allennlp.common.testing import ModelTestCase
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.iterators import BasicIterator
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader, WikiTablesDatasetReader
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.training.callback_trainer import CallbackTrainer
from allennlp.training.callbacks import (
        Events,
        LogToTensorboard, Checkpoint, Validate, PostToUrl, GradientNormAndClip,
        UpdateLearningRate, UpdateMomentum, TrackMetrics, UpdateMovingAverage
)
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import ExponentialMovingAverage
from allennlp.training.tensorboard_writer import TensorboardWriter


class TestCallbackTrainer(ModelTestCase):
    def setUp(self):
        super().setUp()

        # A lot of the tests want access to the metric tracker
        # so we add a property that gets it by grabbing it from
        # the relevant callback.
        def metric_tracker(self: CallbackTrainer):
            for callback in self.handler.callbacks():
                if isinstance(callback, TrackMetrics):
                    return callback.metric_tracker
            return None

        setattr(CallbackTrainer, 'metric_tracker', property(metric_tracker))

        self.instances = SequenceTaggingDatasetReader().read(self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv')
        vocab = Vocabulary.from_instances(self.instances)
        self.vocab = vocab
        self.model_params = Params({
                "text_field_embedder": {
                        "token_embedders": {
                                "tokens": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                        }
                                }
                        },
                "encoder": {
                        "type": "lstm",
                        "input_size": 5,
                        "hidden_size": 7,
                        "num_layers": 2
                        }
                })
        self.model = SimpleTagger.from_params(vocab=self.vocab, params=self.model_params)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9)
        self.iterator = BasicIterator(batch_size=2)
        self.iterator.index_with(vocab)


    def tearDown(self):
        super().tearDown()
        delattr(CallbackTrainer, 'metric_tracker')

    def default_callbacks(self,
                          validation_metric: str = "-loss",
                          patience: int = None,
                          max_checkpoints: int = 20,
                          checkpoint_every: int = None,
                          model_save_interval: float = None,
                          serialization_dir: str = "__DEFAULT__",
                          validation_data: Iterable[Instance] = None,
                          validation_iterator: DataIterator = None,
                          batch_size: int = 2):
        if serialization_dir == "__DEFAULT__":
            serialization_dir = self.TEST_DIR
        checkpointer = Checkpointer(serialization_dir,
                                    checkpoint_every,
                                    max_checkpoints)
        tensorboard = TensorboardWriter(get_batch_num_total=lambda: None)

        if validation_iterator is None:
            validation_iterator = BasicIterator(batch_size=batch_size)
            validation_iterator.index_with(self.vocab)

        return [
                LogToTensorboard(log_batch_size_period=10, tensorboard=tensorboard),
                Checkpoint(checkpointer, model_save_interval),
                Validate(validation_data=self.instances if validation_data is None else validation_data,
                         validation_iterator=validation_iterator),
                TrackMetrics(patience, validation_metric),
                GradientNormAndClip(),
        ]

    def test_end_to_end(self):
        self.ensure_model_can_train_save_and_load(
                self.FIXTURES_ROOT / 'simple_tagger' / 'experiment_callback_trainer.json')

    def test_trainer_can_run_from_params(self):
        # pylint: disable=bad-continuation
        from allennlp.commands.train import train_model

        params = Params({
                "trainer": {
                    "type": "callback",
                    "optimizer": {"type": "sgd", "lr": 0.01, "momentum": 0.9},
                    "num_epochs": 2,
                    "callbacks": [
                        "checkpoint",
                        "track_metrics",
                        "validate",
                        {"type": "log_to_tensorboard", "log_batch_size_period": 10}
                    ]
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": str(self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv'),
                "validation_data_path": str(self.FIXTURES_ROOT / 'data' / 'sequence_tagging.tsv'),
                "model": {
                    "type": "simple_tagger",
                    "text_field_embedder": {
                        "token_embedders": {
                            "tokens": {
                                "type": "embedding",
                                "embedding_dim": 5
                            }
                        }
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": 5,
                        "hidden_size": 7,
                        "num_layers": 2
                    }
                },
                "iterator": {"type": "basic", "batch_size": 2}
        })

        train_model(params, self.TEST_DIR)
        with open(self.TEST_DIR / 'metrics.json') as f:
            metrics = json.load(f)
        assert 'best_validation_loss' in metrics
        assert isinstance(metrics['best_validation_loss'], float)
        assert 'best_validation_accuracy' in metrics
        assert isinstance(metrics['best_validation_accuracy'], float)
        assert 'best_validation_accuracy3' in metrics
        assert isinstance(metrics['best_validation_accuracy3'], float)
        assert 'best_epoch' in metrics
        assert isinstance(metrics['best_epoch'], int)

    def test_trainer_can_run(self):
        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  callbacks=self.default_callbacks(serialization_dir=None),
                                  num_epochs=2)
        metrics = trainer.train()
        assert 'best_validation_loss' in metrics
        assert isinstance(metrics['best_validation_loss'], float)
        assert 'best_validation_accuracy' in metrics
        assert isinstance(metrics['best_validation_accuracy'], float)
        assert 'best_validation_accuracy3' in metrics
        assert isinstance(metrics['best_validation_accuracy3'], float)
        assert 'best_epoch' in metrics
        assert isinstance(metrics['best_epoch'], int)
        assert 'peak_cpu_memory_MB' in metrics

        # Making sure that both increasing and decreasing validation metrics work.
        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  callbacks=self.default_callbacks(validation_metric="+loss",
                                                                   serialization_dir=None),
                                  num_epochs=2)
        metrics = trainer.train()
        assert 'best_validation_loss' in metrics
        assert isinstance(metrics['best_validation_loss'], float)
        assert 'best_validation_accuracy' in metrics
        assert isinstance(metrics['best_validation_accuracy'], float)
        assert 'best_validation_accuracy3' in metrics
        assert isinstance(metrics['best_validation_accuracy3'], float)
        assert 'best_epoch' in metrics
        assert isinstance(metrics['best_epoch'], int)
        assert 'peak_cpu_memory_MB' in metrics
        assert isinstance(metrics['peak_cpu_memory_MB'], float)
        assert metrics['peak_cpu_memory_MB'] > 0

    @responses.activate
    def test_trainer_posts_to_url(self):
        url = 'https://slack.com?webhook=ewifjweoiwjef'
        responses.add(responses.POST, url)
        post_to_url = PostToUrl(url, message="only a test")
        callbacks = self.default_callbacks() + [post_to_url]
        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=2,
                                  callbacks=callbacks)
        trainer.train()

        assert len(responses.calls) == 1
        assert responses.calls[0].response.request.body == b'{"text": "only a test"}'


    def test_trainer_can_run_exponential_moving_average(self):
        moving_average = ExponentialMovingAverage(self.model.named_parameters(), decay=0.9999)
        callbacks = self.default_callbacks() + [UpdateMovingAverage(moving_average)]
        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=2,
                                  callbacks=callbacks)
        trainer.train()

    def test_trainer_can_run_ema_from_params(self):
        uma_params = Params({"moving_average": {"decay": 0.9999}})
        callbacks = self.default_callbacks() + [UpdateMovingAverage.from_params(uma_params, self.model)]
        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=2,
                                  callbacks=callbacks)
        trainer.train()


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    def test_trainer_can_run_cuda(self):
        self.model.cuda()
        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=2,
                                  callbacks=self.default_callbacks(),
                                  cuda_device=0)
        trainer.train()

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason="Need multiple GPUs.")
    def test_trainer_can_run_multiple_gpu(self):
        self.model.cuda()
        class MetaDataCheckWrapper(Model):
            """
            Checks that the metadata field has been correctly split across the batch dimension
            when running on multiple gpus.
            """
            def __init__(self, model):
                super().__init__(model.vocab)
                self.model = model

            def forward(self, **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore # pylint: disable=arguments-differ
                assert 'metadata' in kwargs and 'tags' in kwargs, \
                    f'tokens and metadata must be provided. Got {kwargs.keys()} instead.'
                batch_size = kwargs['tokens']['tokens'].size()[0]
                assert len(kwargs['metadata']) == batch_size, \
                    f'metadata must be split appropriately. Expected {batch_size} elements, ' \
                    f"got {len(kwargs['metadata'])} elements."
                return self.model.forward(**kwargs)

        multigpu_iterator = BasicIterator(batch_size=4)
        multigpu_iterator.index_with(self.vocab)
        trainer = CallbackTrainer(MetaDataCheckWrapper(self.model),
                                  training_data=self.instances,
                                  iterator=multigpu_iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=2,
                                  callbacks=self.default_callbacks(),
                                  cuda_device=[0, 1])
        metrics = trainer.train()
        assert 'peak_cpu_memory_MB' in metrics
        assert isinstance(metrics['peak_cpu_memory_MB'], float)
        assert metrics['peak_cpu_memory_MB'] > 0
        assert 'peak_gpu_0_memory_MB' in metrics
        assert isinstance(metrics['peak_gpu_0_memory_MB'], int)
        assert 'peak_gpu_1_memory_MB' in metrics
        assert isinstance(metrics['peak_gpu_1_memory_MB'], int)

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason="Need multiple GPUs.")
    def test_production_rule_field_with_multiple_gpus(self):
        wikitables_dir = 'allennlp/tests/fixtures/data/wikitables/'
        offline_lf_directory = wikitables_dir + 'action_space_walker_output/'
        wikitables_reader = WikiTablesDatasetReader(tables_directory=wikitables_dir,
                                                    offline_logical_forms_directory=offline_lf_directory)
        instances = wikitables_reader.read(wikitables_dir + 'sample_data.examples')
        archive_path = self.FIXTURES_ROOT / 'semantic_parsing' / 'wikitables' / 'serialization' / 'model.tar.gz'
        model = load_archive(archive_path).model
        model.cuda()

        multigpu_iterator = BasicIterator(batch_size=4)
        multigpu_iterator.index_with(model.vocab)


        trainer = CallbackTrainer(
                model,
                instances,
                multigpu_iterator,
                self.optimizer,
                num_epochs=2,
                cuda_device=[0, 1],
                callbacks=[
                        GradientNormAndClip()
                ])
        trainer.train()

    def test_trainer_can_resume_training(self):
        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  callbacks=self.default_callbacks(),
                                  num_epochs=1, serialization_dir=self.TEST_DIR)
        trainer.train()


        new_trainer = CallbackTrainer(self.model,
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      callbacks=self.default_callbacks(),
                                      num_epochs=3, serialization_dir=self.TEST_DIR)

        new_trainer.handler.fire_event(Events.TRAINING_START)

        assert new_trainer.epoch_number == 1

        tracker = new_trainer.metric_tracker

        assert tracker is not None
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None

        new_trainer.train()

    def test_trainer_can_resume_training_for_exponential_moving_average(self):
        moving_average = ExponentialMovingAverage(self.model.named_parameters())
        callbacks = self.default_callbacks() + [UpdateMovingAverage(moving_average)]

        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=1, serialization_dir=self.TEST_DIR,
                                  callbacks=callbacks)
        trainer.train()

        new_moving_average = ExponentialMovingAverage(self.model.named_parameters())
        new_callbacks = self.default_callbacks() + [UpdateMovingAverage(new_moving_average)]

        new_trainer = CallbackTrainer(self.model,
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      num_epochs=3, serialization_dir=self.TEST_DIR,
                                      callbacks=new_callbacks)

        new_trainer.handler.fire_event(Events.TRAINING_START)  # pylint: disable=protected-access
        assert new_trainer.epoch_number == 1

        tracker = trainer.metric_tracker  # pylint: disable=protected-access
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None  # pylint: disable=protected-access

        new_trainer.train()

    def test_training_metrics_consistent_with_and_without_validation(self):
        default_callbacks = self.default_callbacks(serialization_dir=None)
        default_callbacks_without_validation = [callback for callback in default_callbacks
                                                if not isinstance(callback, Validate)]
        trainer1 = CallbackTrainer(copy.deepcopy(self.model),
                                   training_data=self.instances,
                                   iterator=self.iterator,
                                   optimizer=copy.deepcopy(self.optimizer),
                                   callbacks=default_callbacks_without_validation,
                                   num_epochs=1, serialization_dir=None)

        trainer1.train()

        trainer2 = CallbackTrainer(copy.deepcopy(self.model),
                                   training_data=self.instances,
                                   iterator=self.iterator,
                                   optimizer=copy.deepcopy(self.optimizer),
                                   callbacks=default_callbacks,
                                   num_epochs=1, serialization_dir=None)

        trainer2.train()
        metrics1 = trainer1.train_metrics
        metrics2 = trainer2.train_metrics
        assert metrics1.keys() == metrics2.keys()
        for key in ['accuracy', 'accuracy3', 'loss']:
            np.testing.assert_almost_equal(metrics1[key], metrics2[key])

    def test_validation_metrics_consistent_with_and_without_tracking(self):
        default_callbacks = self.default_callbacks(serialization_dir=None)
        default_callbacks_without_tracking = [callback for callback in default_callbacks
                                              if not isinstance(callback, TrackMetrics)]
        trainer1 = CallbackTrainer(copy.deepcopy(self.model),
                                   training_data=self.instances,
                                   iterator=self.iterator,
                                   optimizer=copy.deepcopy(self.optimizer),
                                   callbacks=default_callbacks_without_tracking,
                                   num_epochs=1, serialization_dir=None)

        trainer1.train()

        trainer2 = CallbackTrainer(copy.deepcopy(self.model),
                                   training_data=self.instances,
                                   iterator=self.iterator,
                                   optimizer=copy.deepcopy(self.optimizer),
                                   callbacks=default_callbacks,
                                   num_epochs=1, serialization_dir=None)

        trainer2.train()
        metrics1 = trainer1.val_metrics
        metrics2 = trainer2.val_metrics
        assert metrics1.keys() == metrics2.keys()
        for key in ['accuracy', 'accuracy3', 'loss']:
            np.testing.assert_almost_equal(metrics1[key], metrics2[key])

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_increasing_metric(
            self):
        new_trainer = CallbackTrainer(self.model,
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      num_epochs=3, serialization_dir=self.TEST_DIR,
                                      callbacks=self.default_callbacks("+test", patience=5))
        tracker = new_trainer.metric_tracker

        # when it is the only metric it should be considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metric(1)
        assert new_tracker.is_best_so_far()

        # when it is the same as one before it it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.3, .3, .3, .2, .5, .1, .3])
        assert not new_tracker.is_best_so_far()

        # when it is the best it is considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.3, .3, .3, .2, .5, .1, 13])
        assert new_tracker.is_best_so_far()

        # when it is not the the best it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.3, .3, .3, .2, .5, .1, .0013])
        assert not new_tracker.is_best_so_far()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_decreasing_metric(self):
        new_trainer = CallbackTrainer(self.model,
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      num_epochs=3, serialization_dir=self.TEST_DIR,
                                      callbacks=self.default_callbacks(patience=5))
        tracker = new_trainer.metric_tracker

        # when it is the only metric it should be considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metric(1)
        assert new_tracker.is_best_so_far()

        # when it is the same as one before it it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.3, .3, .3, .2, .5, .1, .3])
        assert not new_tracker.is_best_so_far()

        # when it is the best it is considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.3, .3, .3, .2, .5, .1, 0.0013])
        assert new_tracker.is_best_so_far()

        # when it is not the the best it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.3, .3, .3, .2, .5, .1, 13])

    def test_should_stop_early_with_increasing_metric(self):
        new_trainer = CallbackTrainer(self.model,
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      num_epochs=3, serialization_dir=self.TEST_DIR,
                                      callbacks=self.default_callbacks(patience=5, validation_metric="+test"))

        tracker = new_trainer.metric_tracker

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.5, .3, .2, .1, .4, .4])
        assert new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.3, .3, .3, .2, .5, .1])
        assert not new_tracker.should_stop_early()


    def test_should_stop_early_with_decreasing_metric(self):
        new_trainer = CallbackTrainer(self.model,
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      num_epochs=3, serialization_dir=self.TEST_DIR,
                                      callbacks=self.default_callbacks(patience=5))
        tracker = new_trainer.metric_tracker

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.02, .3, .2, .1, .4, .4])
        assert new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.3, .3, .2, .1, .4, .5])
        assert not new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([.1, .3, .2, .1, .4, .5])
        assert new_tracker.should_stop_early()

    def test_should_stop_early_with_early_stopping_disabled(self):
        # Increasing metric
        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=100,
                                  callbacks=self.default_callbacks(validation_metric="+test"))
        tracker = trainer.metric_tracker
        tracker.add_metrics([float(i) for i in reversed(range(20))])
        assert not tracker.should_stop_early()

        # Decreasing metric
        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=100,
                                  callbacks=self.default_callbacks(validation_metric="-test"))
        tracker = trainer.metric_tracker
        tracker.add_metrics([float(i) for i in range(20)])
        assert not tracker.should_stop_early()

    def test_should_stop_early_with_invalid_patience(self):
        for patience in [0, -1, -2, 1.5, 'None']:
            with pytest.raises(ConfigurationError):
                CallbackTrainer(self.model,
                                training_data=self.instances,
                                iterator=self.iterator,
                                optimizer=self.optimizer,
                                num_epochs=100,
                                callbacks=self.default_callbacks(patience=patience, validation_metric="+test"))

    def test_trainer_can_run_and_resume_with_momentum_scheduler(self):
        scheduler = MomentumScheduler.from_params(
                self.optimizer, Params({"type": "inverted_triangular", "cool_down": 2, "warm_up": 2}))
        callbacks = self.default_callbacks() + [UpdateMomentum(scheduler)]
        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=4,
                                  callbacks=callbacks,
                                  serialization_dir=self.TEST_DIR)
        trainer.train()

        new_scheduler = MomentumScheduler.from_params(
                self.optimizer, Params({"type": "inverted_triangular", "cool_down": 2, "warm_up": 2}))
        new_callbacks = self.default_callbacks() + [UpdateMomentum(new_scheduler)]
        new_trainer = CallbackTrainer(model=self.model,
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      num_epochs=6,
                                      callbacks=new_callbacks,
                                      serialization_dir=self.TEST_DIR)
        new_trainer.handler.fire_event(Events.TRAINING_START)
        assert new_trainer.epoch_number == 4
        assert new_scheduler.last_epoch == 3
        new_trainer.train()

    def test_trainer_can_run_with_lr_scheduler(self):
        lr_params = Params({"type": "reduce_on_plateau"})
        lr_scheduler = LearningRateScheduler.from_params(self.optimizer, lr_params)
        callbacks = self.default_callbacks() + [UpdateLearningRate(lr_scheduler)]

        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  callbacks=callbacks,
                                  num_epochs=2)
        trainer.train()

    def test_trainer_can_resume_with_lr_scheduler(self):
        lr_scheduler = LearningRateScheduler.from_params(
                self.optimizer, Params({"type": "exponential", "gamma": 0.5}))
        callbacks = self.default_callbacks() + [UpdateLearningRate(lr_scheduler)]

        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  callbacks=callbacks,
                                  num_epochs=2, serialization_dir=self.TEST_DIR)
        trainer.train()

        new_lr_scheduler = LearningRateScheduler.from_params(
                self.optimizer, Params({"type": "exponential", "gamma": 0.5}))
        callbacks = self.default_callbacks() + [UpdateLearningRate(new_lr_scheduler)]

        new_trainer = CallbackTrainer(model=self.model,
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      callbacks=callbacks,
                                      num_epochs=4, serialization_dir=self.TEST_DIR)
        new_trainer.handler.fire_event(Events.TRAINING_START)
        assert new_trainer.epoch_number == 2
        assert new_lr_scheduler.lr_scheduler.last_epoch == 1
        new_trainer.train()

    def test_trainer_raises_on_model_with_no_loss_key(self):
        class FakeModel(Model):
            def forward(self, **kwargs):  # pylint: disable=arguments-differ,unused-argument
                return {}
        with pytest.raises(RuntimeError):
            trainer = CallbackTrainer(FakeModel(None),
                                      training_data=self.instances,
                                      iterator=self.iterator,
                                      optimizer=self.optimizer,
                                      callbacks=self.default_callbacks(),
                                      num_epochs=2, serialization_dir=self.TEST_DIR)
            trainer.train()

    def test_trainer_can_log_histograms(self):
        # enable activation logging
        for module in self.model.modules():
            module.should_log_activations = True

        callbacks = [cb for cb in self.default_callbacks() if not isinstance(cb, LogToTensorboard)]
        # The lambda: None is unfortunate, but it will get replaced by the callback.
        tensorboard = TensorboardWriter(lambda: None, histogram_interval=2)
        callbacks.append(LogToTensorboard(tensorboard))

        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=3,
                                  serialization_dir=self.TEST_DIR,
                                  callbacks=callbacks)
        trainer.train()

    def test_trainer_respects_num_serialized_models_to_keep(self):
        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=5,
                                  serialization_dir=self.TEST_DIR,
                                  callbacks=self.default_callbacks(max_checkpoints=3))
        trainer.train()

        # Now check the serialized files
        for prefix in ['model_state_epoch_*', 'training_state_epoch_*']:
            file_names = glob.glob(os.path.join(self.TEST_DIR, prefix))
            epochs = [int(re.search(r"_([0-9])\.th", fname).group(1))
                      for fname in file_names]
            assert sorted(epochs) == [2, 3, 4]

    def test_trainer_saves_metrics_every_epoch(self):
        trainer = CallbackTrainer(model=self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=5,
                                  serialization_dir=self.TEST_DIR,
                                  callbacks=self.default_callbacks(max_checkpoints=3))
        trainer.train()

        for epoch in range(5):
            epoch_file = self.TEST_DIR / f'metrics_epoch_{epoch}.json'
            assert epoch_file.exists()
            metrics = json.load(open(epoch_file))
            assert "validation_loss" in metrics
            assert "best_validation_loss" in metrics
            assert metrics.get("epoch") == epoch

    def test_trainer_respects_keep_serialized_model_every_num_seconds(self):
        # To test:
        #   Create an iterator that sleeps for 2.5 second per epoch, so the total training
        #       time for one epoch is slightly greater then 2.5 seconds.
        #   Run for 6 epochs, keeping the last 2 models, models also kept every 5 seconds.
        #   Check the resulting checkpoints.  Should then have models at epochs
        #       2, 4, plus the last two at 5 and 6.
        class WaitingIterator(BasicIterator):
            # pylint: disable=arguments-differ
            def _create_batches(self, *args, **kwargs):
                time.sleep(2.5)
                return super(WaitingIterator, self)._create_batches(*args, **kwargs)

        waiting_iterator = WaitingIterator(batch_size=2)
        waiting_iterator.index_with(self.vocab)

        # Don't want validation iterator to wait.
        viterator = BasicIterator(batch_size=2)
        viterator.index_with(self.vocab)

        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=waiting_iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=6,
                                  serialization_dir=self.TEST_DIR,
                                  callbacks=self.default_callbacks(max_checkpoints=2,
                                                                   checkpoint_every=5,
                                                                   validation_iterator=viterator))
        trainer.train()

        # Now check the serialized files
        for prefix in ['model_state_epoch_*', 'training_state_epoch_*']:
            file_names = glob.glob(os.path.join(self.TEST_DIR, prefix))
            epochs = [int(re.search(r"_([0-9])\.th", fname).group(1))
                      for fname in file_names]
            # epoch N has N-1 in file name
            assert sorted(epochs) == [1, 3, 4, 5]

    def test_trainer_can_log_learning_rates_tensorboard(self):
        callbacks = [cb for cb in self.default_callbacks() if not isinstance(cb, LogToTensorboard)]
        # The lambda: None is unfortunate, but it will get replaced by the callback.
        tensorboard = TensorboardWriter(lambda: None, should_log_learning_rate=True, summary_interval=2)
        callbacks.append(LogToTensorboard(tensorboard))

        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=self.iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=2,
                                  serialization_dir=self.TEST_DIR,
                                  callbacks=callbacks)

        trainer.train()

    def test_trainer_saves_models_at_specified_interval(self):
        iterator = BasicIterator(batch_size=4)
        iterator.index_with(self.vocab)

        trainer = CallbackTrainer(self.model,
                                  training_data=self.instances,
                                  iterator=iterator,
                                  optimizer=self.optimizer,
                                  num_epochs=2,
                                  serialization_dir=self.TEST_DIR,
                                  callbacks=self.default_callbacks(model_save_interval=0.0001))

        trainer.train()

        # Now check the serialized files for models saved during the epoch.
        prefix = 'model_state_epoch_*'
        file_names = sorted(glob.glob(os.path.join(self.TEST_DIR, prefix)))
        epochs = [re.search(r"_([0-9\.\-]+)\.th", fname).group(1)
                  for fname in file_names]
        # We should have checkpoints at the end of each epoch and during each, e.g.
        # [0.timestamp, 0, 1.timestamp, 1]
        assert len(epochs) == 4
        assert epochs[3] == '1'
        assert '.' in epochs[0]

        # Now make certain we can restore from timestamped checkpoint.
        # To do so, remove the checkpoint from the end of epoch 1&2, so
        # that we are forced to restore from the timestamped checkpoints.
        for k in range(2):
            os.remove(os.path.join(self.TEST_DIR, 'model_state_epoch_{}.th'.format(k)))
            os.remove(os.path.join(self.TEST_DIR, 'training_state_epoch_{}.th'.format(k)))
        os.remove(os.path.join(self.TEST_DIR, 'best.th'))

        restore_trainer = CallbackTrainer(
                self.model,
                training_data=self.instances,
                iterator=iterator,
                optimizer=self.optimizer,
                num_epochs=2,
                serialization_dir=self.TEST_DIR,
                callbacks=self.default_callbacks(model_save_interval=0.0001))
        restore_trainer.handler.fire_event(Events.TRAINING_START)
        assert restore_trainer.epoch_number == 2
        # One batch per epoch.
        assert restore_trainer.batch_num_total == 2

    def test_trainer_saves_and_loads_best_validation_metrics_correctly_1(self):
        # Use -loss and run 1 epoch of original-training, and one of restored-training
        # Run 1 epoch of original training.
        trainer = CallbackTrainer(
                self.model,
                training_data=self.instances,
                iterator=self.iterator,
                optimizer=self.optimizer,
                callbacks=self.default_callbacks(),
                num_epochs=1, serialization_dir=self.TEST_DIR)
        trainer.train()
        _ = trainer.handler.fire_event(Events.TRAINING_START)
        best_epoch_1 = trainer.metric_tracker.best_epoch
        best_validation_metrics_epoch_1 = trainer.metric_tracker.best_epoch_metrics
        # best_validation_metrics_epoch_1: {'accuracy': 0.75, 'accuracy3': 1.0, 'loss': 0.6243013441562653}
        assert isinstance(best_validation_metrics_epoch_1, dict)
        assert "loss" in best_validation_metrics_epoch_1

        # Run 1 epoch of restored training.
        restore_trainer = CallbackTrainer(
                self.model,
                training_data=self.instances,
                iterator=self.iterator,
                optimizer=self.optimizer,
                callbacks=self.default_callbacks(),
                num_epochs=2, serialization_dir=self.TEST_DIR)
        restore_trainer.train()
        _ = restore_trainer.handler.fire_event(Events.TRAINING_START)
        best_epoch_2 = restore_trainer.metric_tracker.best_epoch
        best_validation_metrics_epoch_2 = restore_trainer.metric_tracker.best_epoch_metrics

        # Because of using -loss, 2nd epoch would be better than 1st. So best val metrics should not be same.
        assert best_epoch_1 == 0 and best_epoch_2 == 1
        assert best_validation_metrics_epoch_2 != best_validation_metrics_epoch_1

    def test_trainer_saves_and_loads_best_validation_metrics_correctly_2(self):
        # Use -loss and run 1 epoch of original-training, and one of restored-training
        # Run 1 epoch of original training.
        trainer = CallbackTrainer(
                self.model,
                training_data=self.instances,
                iterator=self.iterator,
                optimizer=self.optimizer,
                callbacks=self.default_callbacks(validation_metric="+loss"),
                num_epochs=1, serialization_dir=self.TEST_DIR)
        trainer.handler.verbose = True
        trainer.train()

        _ = trainer.handler.fire_event(Events.TRAINING_START)
        best_epoch_1 = trainer.metric_tracker.best_epoch
        best_validation_metrics_epoch_1 = trainer.metric_tracker.best_epoch_metrics
        # best_validation_metrics_epoch_1: {'accuracy': 0.75, 'accuracy3': 1.0, 'loss': 0.6243013441562653}
        assert isinstance(best_validation_metrics_epoch_1, dict)
        assert "loss" in best_validation_metrics_epoch_1

        # Run 1 more epoch of restored training.
        restore_trainer = CallbackTrainer(
                self.model,
                training_data=self.instances,
                iterator=self.iterator,
                optimizer=self.optimizer,
                callbacks=self.default_callbacks(validation_metric="+loss"),
                num_epochs=2, serialization_dir=self.TEST_DIR)
        print("restore trainer")
        restore_trainer.handler.verbose = True
        restore_trainer.train()
        _ = restore_trainer.handler.fire_event(Events.TRAINING_START)
        best_epoch_2 = restore_trainer.metric_tracker.best_epoch
        best_validation_metrics_epoch_2 = restore_trainer.metric_tracker.best_epoch_metrics

        # Because of using +loss, 2nd epoch won't be better than 1st. So best val metrics should be same.
        assert best_epoch_1 == best_epoch_2 == 0
        assert best_validation_metrics_epoch_2 == best_validation_metrics_epoch_1

    def test_restored_training_returns_best_epoch_metrics_even_if_no_better_epoch_is_found_after_restoring(self):
        # Instead of -loss, use +loss to assure 2nd epoch is considered worse.
        # Run 1 epoch of original training.
        original_trainer = CallbackTrainer(self.model,
                                           training_data=self.instances,
                                           iterator=self.iterator,
                                           optimizer=self.optimizer,
                                           callbacks=self.default_callbacks(validation_metric="+loss"),
                                           num_epochs=1, serialization_dir=self.TEST_DIR)
        training_metrics = original_trainer.train()

        # Run 1 epoch of restored training.
        restored_trainer = CallbackTrainer(
                self.model,
                training_data=self.instances,
                iterator=self.iterator,
                optimizer=self.optimizer,
                callbacks=self.default_callbacks(validation_metric="+loss"),
                num_epochs=2, serialization_dir=self.TEST_DIR)
        restored_metrics = restored_trainer.train()

        assert "best_validation_loss" in restored_metrics
        assert "best_validation_accuracy" in restored_metrics
        assert "best_validation_accuracy3" in restored_metrics
        assert "best_epoch" in restored_metrics

        # Epoch 2 validation loss should be lesser than that of Epoch 1
        assert training_metrics["best_validation_loss"] == restored_metrics["best_validation_loss"]
        assert training_metrics["best_epoch"] == 0
        assert training_metrics["validation_loss"] > restored_metrics["validation_loss"]
