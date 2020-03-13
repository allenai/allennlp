import copy
import glob
import json
import os
import re
import time

import math
import pytest

try:
    from apex import amp
except ImportError:
    amp = None
import torch
from torch.utils.data import DataLoader

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.models.model import Model
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import CosineWithRestarts
from allennlp.training.learning_rate_schedulers import ExponentialLearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import ExponentialMovingAverage
from allennlp.training.util import sparse_clip_norm
from allennlp.data import allennlp_collate


class TrainerTestBase(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.instances = SequenceTaggingDatasetReader().read(
            self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"
        )
        vocab = Vocabulary.from_instances(self.instances)
        self.vocab = vocab
        self.model_params = Params(
            {
                "text_field_embedder": {
                    "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                },
                "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
            }
        )
        self.model = SimpleTagger.from_params(vocab=self.vocab, params=self.model_params)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9)
        self.data_loader = DataLoader(self.instances, batch_size=2, collate_fn=allennlp_collate)
        self.validation_data_loader = DataLoader(
            self.instances, batch_size=2, collate_fn=allennlp_collate
        )
        self.instances.index_with(vocab)


class TestTrainer(TrainerTestBase):
    def test_trainer_can_run(self):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
        )
        metrics = trainer.train()
        assert "best_validation_loss" in metrics
        assert isinstance(metrics["best_validation_loss"], float)
        assert "best_validation_accuracy" in metrics
        assert isinstance(metrics["best_validation_accuracy"], float)
        assert "best_validation_accuracy3" in metrics
        assert isinstance(metrics["best_validation_accuracy3"], float)
        assert "best_epoch" in metrics
        assert isinstance(metrics["best_epoch"], int)

        # Making sure that both increasing and decreasing validation metrics work.
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=2,
        )
        metrics = trainer.train()
        assert "best_validation_loss" in metrics
        assert isinstance(metrics["best_validation_loss"], float)
        assert "best_validation_accuracy" in metrics
        assert isinstance(metrics["best_validation_accuracy"], float)
        assert "best_validation_accuracy3" in metrics
        assert isinstance(metrics["best_validation_accuracy3"], float)
        assert "best_epoch" in metrics
        assert isinstance(metrics["best_epoch"], int)
        assert "peak_cpu_memory_MB" in metrics
        assert isinstance(metrics["peak_cpu_memory_MB"], float)
        assert metrics["peak_cpu_memory_MB"] > 0

    def test_trainer_can_run_exponential_moving_average(self):
        moving_average = ExponentialMovingAverage(self.model.named_parameters(), decay=0.9999)
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
            moving_average=moving_average,
        )
        trainer.train()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    def test_trainer_can_run_cuda(self):
        self.model.cuda()
        trainer = Trainer(self.model, self.optimizer, self.data_loader, num_epochs=2, cuda_device=0)
        metrics = trainer.train()
        assert "peak_cpu_memory_MB" in metrics
        assert isinstance(metrics["peak_cpu_memory_MB"], float)
        assert metrics["peak_cpu_memory_MB"] > 0
        assert "peak_gpu_0_memory_MB" in metrics
        assert isinstance(metrics["peak_gpu_0_memory_MB"], int)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 or more GPUs required.")
    def test_passing_trainer_multiple_gpus_raises_error(self):
        self.model.cuda()

        with pytest.raises(ConfigurationError):
            Trainer(
                self.model, self.optimizer, self.data_loader, num_epochs=2, cuda_device=[0, 1],
            )

    def test_trainer_can_resume_training(self):
        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
        )
        trainer.train()
        new_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
        )

        epoch = new_trainer._restore_checkpoint()
        assert epoch == 1

        tracker = trainer._metric_tracker
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None

        new_trainer.train()

    def test_trainer_can_resume_training_for_exponential_moving_average(self):
        moving_average = ExponentialMovingAverage(self.model.named_parameters())

        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
            moving_average=moving_average,
        )
        trainer.train()

        new_moving_average = ExponentialMovingAverage(self.model.named_parameters())
        new_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            moving_average=new_moving_average,
        )

        epoch = new_trainer._restore_checkpoint()
        assert epoch == 1

        tracker = trainer._metric_tracker
        assert tracker.is_best_so_far()
        assert tracker._best_so_far is not None

        new_trainer.train()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_increasing_metric(
        self,
    ):
        new_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="+test",
        )
        tracker = new_trainer._metric_tracker

        # when it is the only metric it should be considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metric(1)
        assert new_tracker.is_best_so_far()

        # when it is the same as one before it it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.3])
        assert not new_tracker.is_best_so_far()

        # when it is the best it is considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 13])
        assert new_tracker.is_best_so_far()

        # when it is not the the best it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.0013])
        assert not new_tracker.is_best_so_far()

    def test_metric_only_considered_best_so_far_when_strictly_better_than_those_before_it_decreasing_metric(
        self,
    ):
        new_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="-test",
        )
        tracker = new_trainer._metric_tracker

        # when it is the only metric it should be considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metric(1)
        assert new_tracker.is_best_so_far()

        # when it is the same as one before it it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.3])
        assert not new_tracker.is_best_so_far()

        # when it is the best it is considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 0.0013])
        assert new_tracker.is_best_so_far()

        # when it is not the the best it is not considered the best
        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.3, 0.3, 0.3, 0.2, 0.5, 0.1, 13])

    def test_should_stop_early_with_increasing_metric(self):
        new_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="+test",
        )

        tracker = new_trainer._metric_tracker

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.5, 0.3, 0.2, 0.1, 0.4, 0.4])
        assert new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.3, 0.3, 0.3, 0.2, 0.5, 0.1])
        assert not new_tracker.should_stop_early()

    def test_should_stop_early_with_flat_lining_metric(self):
        flatline = [0.2] * 6
        tracker = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="+test",
        )._metric_tracker
        tracker.add_metrics(flatline)
        assert tracker.should_stop_early

        tracker = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="-test",
        )._metric_tracker
        tracker.add_metrics(flatline)
        assert tracker.should_stop_early

    def test_should_stop_early_with_decreasing_metric(self):
        new_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            patience=5,
            validation_metric="-test",
        )
        tracker = new_trainer._metric_tracker

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.02, 0.3, 0.2, 0.1, 0.4, 0.4])
        assert new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.3, 0.3, 0.2, 0.1, 0.4, 0.5])
        assert not new_tracker.should_stop_early()

        new_tracker = copy.deepcopy(tracker)
        new_tracker.add_metrics([0.1, 0.3, 0.2, 0.1, 0.4, 0.5])
        assert new_tracker.should_stop_early()

    def test_should_stop_early_with_early_stopping_disabled(self):
        # Increasing metric
        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=100,
            patience=None,
            validation_metric="+test",
        )
        tracker = trainer._metric_tracker
        tracker.add_metrics([float(i) for i in reversed(range(20))])
        assert not tracker.should_stop_early()

        # Decreasing metric
        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=100,
            patience=None,
            validation_metric="-test",
        )
        tracker = trainer._metric_tracker
        tracker.add_metrics([float(i) for i in range(20)])
        assert not tracker.should_stop_early()

    def test_should_stop_early_with_invalid_patience(self):
        for patience in [0, -1, -2, 1.5, "None"]:
            with pytest.raises(
                ConfigurationError,
                match='.* is an invalid value for "patience": '
                "it must be a positive integer or None "
                "\\(if you want to disable early stopping\\)",
            ):
                Trainer(
                    self.model,
                    self.optimizer,
                    self.data_loader,
                    validation_data_loader=self.validation_data_loader,
                    num_epochs=100,
                    patience=patience,
                    validation_metric="+test",
                )

    def test_trainer_can_run_and_resume_with_momentum_scheduler(self):
        scheduler = MomentumScheduler.from_params(
            optimizer=self.optimizer,
            params=Params({"type": "inverted_triangular", "cool_down": 2, "warm_up": 2}),
        )
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            momentum_scheduler=scheduler,
            validation_metric="-loss",
            validation_data_loader=self.validation_data_loader,
            num_epochs=4,
            serialization_dir=self.TEST_DIR,
        )
        trainer.train()

        new_scheduler = MomentumScheduler.from_params(
            optimizer=self.optimizer,
            params=Params({"type": "inverted_triangular", "cool_down": 2, "warm_up": 2}),
        )
        new_trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            momentum_scheduler=new_scheduler,
            validation_metric="-loss",
            validation_data_loader=self.validation_data_loader,
            num_epochs=6,
            serialization_dir=self.TEST_DIR,
        )
        epoch = new_trainer._restore_checkpoint()
        assert epoch == 4
        assert new_trainer._momentum_scheduler.last_epoch == 3
        new_trainer.train()

    def test_trainer_can_run_with_lr_scheduler(self):
        lr_scheduler = ExponentialLearningRateScheduler(self.optimizer, gamma=0.5)
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            learning_rate_scheduler=lr_scheduler,
            validation_metric="-loss",
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
        )
        trainer.train()

    def test_trainer_can_resume_with_lr_scheduler(self):
        lr_scheduler = CosineWithRestarts(self.optimizer, t_initial=5)
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            learning_rate_scheduler=lr_scheduler,
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
        )
        trainer.train()

        new_lr_scheduler = CosineWithRestarts(self.optimizer, t_initial=5)
        new_trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            learning_rate_scheduler=new_lr_scheduler,
            validation_data_loader=self.validation_data_loader,
            num_epochs=4,
            serialization_dir=self.TEST_DIR,
        )
        epoch = new_trainer._restore_checkpoint()
        assert epoch == 2
        assert new_trainer._learning_rate_scheduler.last_epoch == 1
        new_trainer.train()

    def test_trainer_raises_on_model_with_no_loss_key(self):
        class FakeModel(Model):
            def forward(self, **kwargs):
                return {}

        with pytest.raises(RuntimeError):
            trainer = Trainer(
                FakeModel(None),
                self.optimizer,
                self.data_loader,
                num_epochs=2,
                serialization_dir=self.TEST_DIR,
            )
            trainer.train()

    def test_trainer_can_log_histograms(self):
        # enable activation logging
        for module in self.model.modules():
            module.should_log_activations = True

        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
            histogram_interval=2,
        )
        trainer.train()

    def test_trainer_respects_num_serialized_models_to_keep(self):
        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=5,
            serialization_dir=self.TEST_DIR,
            num_serialized_models_to_keep=3,
        )
        trainer.train()

        # Now check the serialized files
        for prefix in ["model_state_epoch_*", "training_state_epoch_*"]:
            file_names = glob.glob(os.path.join(self.TEST_DIR, prefix))
            epochs = [int(re.search(r"_([0-9])\.th", fname).group(1)) for fname in file_names]
            assert sorted(epochs) == [2, 3, 4]

    def test_trainer_saves_metrics_every_epoch(self):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            data_loader=self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=5,
            serialization_dir=self.TEST_DIR,
            num_serialized_models_to_keep=3,
        )
        trainer.train()

        for epoch in range(5):
            epoch_file = self.TEST_DIR / f"metrics_epoch_{epoch}.json"
            assert epoch_file.exists()
            metrics = json.load(open(epoch_file))
            assert "validation_loss" in metrics
            assert "best_validation_loss" in metrics
            assert metrics.get("epoch") == epoch

    def test_trainer_respects_keep_serialized_model_every_num_seconds(self):
        # To test:
        #   Create an fake data loader that sleeps for 2.5 second per epoch, so the total
        #   training time for one epoch is slightly greater then 2.5 seconds.
        #   Run for 6 epochs, keeping the last 2 models, models also kept every 5 seconds.
        #   Check the resulting checkpoints.  Should then have models at epochs
        #       2, 4, plus the last two at 5 and 6.

        class SlowDataLoader:
            data_loader = DataLoader(self.instances, batch_size=2, collate_fn=allennlp_collate)

            def __iter__(self):
                time.sleep(2.5)
                return iter(self.data_loader)

            def __len__(self):
                return len(self.data_loader)

        trainer = Trainer(
            self.model,
            self.optimizer,
            SlowDataLoader(),
            num_epochs=6,
            serialization_dir=self.TEST_DIR,
            num_serialized_models_to_keep=2,
            keep_serialized_model_every_num_seconds=5,
        )
        trainer.train()

        # Now check the serialized files
        for prefix in ["model_state_epoch_*", "training_state_epoch_*"]:
            file_names = glob.glob(os.path.join(self.TEST_DIR, prefix))
            epochs = [int(re.search(r"_([0-9])\.th", fname).group(1)) for fname in file_names]
            # epoch N has N-1 in file name
            assert sorted(epochs) == [1, 3, 4, 5]

    def test_trainer_can_log_learning_rates_tensorboard(self):
        data_loader = DataLoader(self.instances, batch_size=4, collate_fn=allennlp_collate)
        trainer = Trainer(
            self.model,
            self.optimizer,
            data_loader,
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            should_log_learning_rate=True,
            summary_interval=2,
        )

        trainer.train()

    def test_trainer_saves_models_at_specified_interval(self):
        data_loader = DataLoader(self.instances, batch_size=4, collate_fn=allennlp_collate)

        trainer = Trainer(
            self.model,
            self.optimizer,
            data_loader,
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            model_save_interval=0.0001,
        )

        trainer.train()

        # Now check the serialized files for models saved during the epoch.
        prefix = "model_state_epoch_*"
        file_names = sorted(glob.glob(os.path.join(self.TEST_DIR, prefix)))
        epochs = [re.search(r"_([0-9\.\-]+)\.th", fname).group(1) for fname in file_names]
        # We should have checkpoints at the end of each epoch and during each, e.g.
        # [0.timestamp, 0, 1.timestamp, 1]
        assert len(epochs) == 4
        assert epochs[3] == "1"
        assert "." in epochs[0]

        # Now make certain we can restore from timestamped checkpoint.
        # To do so, remove the checkpoint from the end of epoch 1&2, so
        # that we are forced to restore from the timestamped checkpoints.
        for k in range(2):
            os.remove(os.path.join(self.TEST_DIR, "model_state_epoch_{}.th".format(k)))
            os.remove(os.path.join(self.TEST_DIR, "training_state_epoch_{}.th".format(k)))
        os.remove(os.path.join(self.TEST_DIR, "best.th"))

        restore_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
            model_save_interval=0.0001,
        )
        epoch = restore_trainer._restore_checkpoint()
        assert epoch == 2
        # One batch per epoch.
        assert restore_trainer._batch_num_total == 2

    def test_trainer_saves_and_loads_best_validation_metrics_correctly_1(self):
        # Use -loss and run 1 epoch of original-training, and one of restored-training
        # Run 1 epoch of original training.
        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="-loss",
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
        )
        trainer.train()
        _ = trainer._restore_checkpoint()
        best_epoch_1 = trainer._metric_tracker.best_epoch
        best_validation_metrics_epoch_1 = trainer._metric_tracker.best_epoch_metrics
        # best_validation_metrics_epoch_1: {'accuracy': 0.75, 'accuracy3': 1.0, 'loss': 0.6243013441562653}
        assert isinstance(best_validation_metrics_epoch_1, dict)
        assert "loss" in best_validation_metrics_epoch_1

        # Run 1 epoch of restored training.
        restore_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="-loss",
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
        )
        restore_trainer.train()
        _ = restore_trainer._restore_checkpoint()
        best_epoch_2 = restore_trainer._metric_tracker.best_epoch
        best_validation_metrics_epoch_2 = restore_trainer._metric_tracker.best_epoch_metrics

        # Because of using -loss, 2nd epoch would be better than 1st. So best val metrics should not be same.
        assert best_epoch_1 == 0 and best_epoch_2 == 1
        assert best_validation_metrics_epoch_2 != best_validation_metrics_epoch_1

    def test_trainer_saves_and_loads_best_validation_metrics_correctly_2(self):
        # Use -loss and run 1 epoch of original-training, and one of restored-training
        # Run 1 epoch of original training.
        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
        )
        trainer.train()

        _ = trainer._restore_checkpoint()
        best_epoch_1 = trainer._metric_tracker.best_epoch
        best_validation_metrics_epoch_1 = trainer._metric_tracker.best_epoch_metrics
        # best_validation_metrics_epoch_1: {'accuracy': 0.75, 'accuracy3': 1.0, 'loss': 0.6243013441562653}
        assert isinstance(best_validation_metrics_epoch_1, dict)
        assert "loss" in best_validation_metrics_epoch_1

        # Run 1 more epoch of restored training.
        restore_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
        )
        restore_trainer.train()
        _ = restore_trainer._restore_checkpoint()
        best_epoch_2 = restore_trainer._metric_tracker.best_epoch
        best_validation_metrics_epoch_2 = restore_trainer._metric_tracker.best_epoch_metrics

        # Because of using +loss, 2nd epoch won't be better than 1st. So best val metrics should be same.
        assert best_epoch_1 == best_epoch_2 == 0
        assert best_validation_metrics_epoch_2 == best_validation_metrics_epoch_1

    def test_restored_training_returns_best_epoch_metrics_even_if_no_better_epoch_is_found_after_restoring(
        self,
    ):
        # Instead of -loss, use +loss to assure 2nd epoch is considered worse.
        # Run 1 epoch of original training.
        original_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=1,
            serialization_dir=self.TEST_DIR,
        )
        training_metrics = original_trainer.train()

        # Run 1 epoch of restored training.
        restored_trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            validation_metric="+loss",
            num_epochs=2,
            serialization_dir=self.TEST_DIR,
        )
        restored_metrics = restored_trainer.train()

        assert "best_validation_loss" in restored_metrics
        assert "best_validation_accuracy" in restored_metrics
        assert "best_validation_accuracy3" in restored_metrics
        assert "best_epoch" in restored_metrics

        # Epoch 2 validation loss should be lesser than that of Epoch 1
        assert training_metrics["best_validation_loss"] == restored_metrics["best_validation_loss"]
        assert training_metrics["best_epoch"] == 0
        assert training_metrics["validation_loss"] > restored_metrics["validation_loss"]

    def test_restoring_works_with_older_checkpointing(self):
        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=3,
            serialization_dir=self.TEST_DIR,
        )
        trainer.train()

        for index in range(3):
            path = str(self.TEST_DIR / "training_state_epoch_{}.th".format(index))
            state = torch.load(path)
            state.pop("metric_tracker")
            state.pop("batch_num_total")
            state["val_metric_per_epoch"] = [0.4, 0.1, 0.8]
            torch.save(state, path)

        next_epoch = trainer._restore_checkpoint()
        best_epoch = trainer._metric_tracker.best_epoch

        # Loss decreases in 3 epochs, but because we hard fed the val metrics as above:
        assert next_epoch == 3
        assert best_epoch == 1
        assert trainer._metric_tracker._best_so_far == 0.1
        assert trainer._metric_tracker._epochs_with_no_improvement == 1

    def test_trainer_can_run_gradient_accumulation(self):
        instances = list(self.instances)
        steps_to_accumulate = 2

        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            validation_data_loader=self.validation_data_loader,
            num_epochs=2,
            num_gradient_accumulation_steps=steps_to_accumulate,
        )
        assert trainer._num_gradient_accumulation_steps == steps_to_accumulate

        metrics = trainer.train()

        num_batches_trained_per_epoch = trainer._batch_num_total // (metrics["training_epochs"] + 1)
        num_batches_expected = math.ceil(
            math.ceil(len(instances) / self.data_loader.batch_size) / steps_to_accumulate
        )

        assert num_batches_trained_per_epoch == num_batches_expected


class TestApexTrainer(TrainerTestBase):
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    @pytest.mark.skipif(amp is None, reason="Apex is not installed.")
    @pytest.mark.spawn
    def test_trainer_can_run_amp(self):
        self.model.cuda()
        trainer = Trainer(
            self.model,
            self.optimizer,
            self.data_loader,
            num_epochs=2,
            cuda_device=0,
            opt_level="O1",
        )
        _ = trainer.train()


class TestSparseClipGrad(AllenNlpTestCase):
    def test_sparse_clip_grad(self):
        # create a sparse embedding layer, then take gradient
        embedding = torch.nn.Embedding(100, 16, sparse=True)
        embedding.zero_grad()
        ids = (torch.rand(17) * 100).long()
        # Set some of the ids to the same value so that the sparse gradient
        # has repeated indices.  This tests some additional logic.
        ids[:5] = 5
        loss = embedding(ids).sum()
        loss.backward()
        assert embedding.weight.grad.is_sparse

        # Now try to clip the gradients.
        _ = sparse_clip_norm([embedding.weight], 1.5)
        # Final norm should be 1.5
        grad = embedding.weight.grad.coalesce()
        self.assertAlmostEqual(grad._values().norm(2.0).item(), 1.5, places=5)
