# pylint: disable=invalid-name
import glob
import os
import re
import time

import torch
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.trainer import Trainer, sparse_clip_norm, is_sparse
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.iterators import BasicIterator
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader


class TestTrainer(AllenNlpTestCase):
    def setUp(self):
        super(TestTrainer, self).setUp()
        self.instances = SequenceTaggingDatasetReader().read('tests/fixtures/data/sequence_tagging.tsv')
        vocab = Vocabulary.from_instances(self.instances)
        self.vocab = vocab
        self.model_params = Params({
                "text_field_embedder": {
                        "tokens": {
                                "type": "embedding",
                                "embedding_dim": 5
                                }
                        },
                "stacked_encoder": {
                        "type": "lstm",
                        "input_size": 5,
                        "hidden_size": 7,
                        "num_layers": 2
                        }
                })
        self.model = SimpleTagger.from_params(self.vocab, self.model_params)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01)
        self.iterator = BasicIterator(batch_size=2)
        self.iterator.index_with(vocab)

    def test_trainer_can_run(self):
        trainer = Trainer(self.model, self.optimizer,
                          self.iterator, self.instances, num_epochs=2)
        trainer.train()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    def test_trainer_can_run_cuda(self):
        trainer = Trainer(self.model, self.optimizer,
                          self.iterator, self.instances, num_epochs=2,
                          cuda_device=0)
        trainer.train()

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason="Need multiple GPUs.")
    def test_trainer_can_run_multiple_gpu(self):
        trainer = Trainer(self.model, self.optimizer,
                          BasicIterator(batch_size=4), self.instances, num_epochs=2,
                          cuda_device=[0, 1])
        trainer.train()

    def test_trainer_can_resume_training(self):
        trainer = Trainer(self.model, self.optimizer,
                          self.iterator, self.instances,
                          validation_dataset=self.instances,
                          num_epochs=1, serialization_dir=self.TEST_DIR)
        trainer.train()
        new_trainer = Trainer(self.model, self.optimizer,
                              self.iterator, self.instances,
                              validation_dataset=self.instances,
                              num_epochs=3, serialization_dir=self.TEST_DIR)

        epoch, val_metrics_per_epoch = new_trainer._restore_checkpoint()  # pylint: disable=protected-access
        assert epoch == 1
        assert len(val_metrics_per_epoch) == 1
        assert isinstance(val_metrics_per_epoch[0], float)
        assert val_metrics_per_epoch[0] != 0.
        new_trainer.train()

    def test_should_stop_early_with_increasing_metric(self):
        new_trainer = Trainer(self.model, self.optimizer,
                              self.iterator, self.instances,
                              validation_dataset=self.instances,
                              num_epochs=3, serialization_dir=self.TEST_DIR,
                              patience=5, validation_metric="+test")
        assert new_trainer._should_stop_early([.5, .3, .2, .1, .4, .4]) #pylint: disable=protected-access
        assert not new_trainer._should_stop_early([.3, .3, .3, .2, .5, .1]) #pylint: disable=protected-access

    def test_should_stop_early_with_decreasing_metric(self):
        new_trainer = Trainer(self.model, self.optimizer,
                              self.iterator, self.instances,
                              validation_dataset=self.instances,
                              num_epochs=3, serialization_dir=self.TEST_DIR,
                              patience=5, validation_metric="-test")
        assert new_trainer._should_stop_early([.02, .3, .2, .1, .4, .4]) #pylint: disable=protected-access
        assert not new_trainer._should_stop_early([.3, .3, .2, .1, .4, .5]) #pylint: disable=protected-access


    def test_train_driver_raises_on_model_with_no_loss_key(self):

        class FakeModel(torch.nn.Module):
            def forward(self, **kwargs):  # pylint: disable=arguments-differ,unused-argument
                return {}
        with pytest.raises(ConfigurationError):
            trainer = Trainer(FakeModel(), self.optimizer,
                              self.iterator, self.instances,
                              num_epochs=2, serialization_dir=self.TEST_DIR)
            trainer.train()

    @pytest.mark.skipif(os.uname().sysname == 'Darwin', reason="Tensorboard logging broken on mac.")
    def test_trainer_can_log_histograms(self):
        # enable activation logging
        for module in self.model.modules():
            module.should_log_activations = True

        trainer = Trainer(self.model, self.optimizer,
                          self.iterator, self.instances, num_epochs=3,
                          serialization_dir=self.TEST_DIR,
                          histogram_interval=2)
        trainer.train()

    def test_trainer_respects_num_serialized_models_to_keep(self):
        trainer = Trainer(self.model, self.optimizer,
                          self.iterator, self.instances, num_epochs=5,
                          serialization_dir=self.TEST_DIR,
                          num_serialized_models_to_keep=3)
        trainer.train()

        # Now check the serialized files
        for prefix in ['model_state_epoch_*', 'training_state_epoch_*']:
            file_names = glob.glob(os.path.join(self.TEST_DIR, prefix))
            epochs = [int(re.search(r"_([0-9])\.th", fname).group(1))
                      for fname in file_names]
            assert sorted(epochs) == [2, 3, 4]

    def test_trainer_respects_keep_serialized_model_every_num_seconds(self):
        # To test:
        #   Create an iterator that sleeps for 0.5 second per epoch, so the total training
        #       time for one epoch is slightly greater then 0.5 seconds.
        #   Run for 6 epochs, keeping the last 2 models, models also kept every 1 second.
        #   Check the resulting checkpoints.  Should then have models at epochs
        #       2, 4, plus the last two at 5 and 6.
        class WaitingIterator(BasicIterator):
            # pylint: disable=arguments-differ
            def _create_batches(self, *args, **kwargs):
                time.sleep(0.5)
                return super(WaitingIterator, self)._create_batches(*args, **kwargs)

        iterator = WaitingIterator(batch_size=2)
        iterator.index_with(self.vocab)

        trainer = Trainer(self.model, self.optimizer,
                          iterator, self.instances, num_epochs=6,
                          serialization_dir=self.TEST_DIR,
                          num_serialized_models_to_keep=2,
                          keep_serialized_model_every_num_seconds=1)
        trainer.train()

        # Now check the serialized files
        for prefix in ['model_state_epoch_*', 'training_state_epoch_*']:
            file_names = glob.glob(os.path.join(self.TEST_DIR, prefix))
            epochs = [int(re.search(r"_([0-9])\.th", fname).group(1))
                      for fname in file_names]
            # epoch N has N-1 in file name
            assert sorted(epochs) == [1, 3, 4, 5]

    def test_trainer_saves_models_at_specified_interval(self):
        iterator = BasicIterator(batch_size=4)
        iterator.index_with(self.vocab)

        trainer = Trainer(self.model, self.optimizer,
                          iterator, self.instances, num_epochs=2,
                          serialization_dir=self.TEST_DIR,
                          model_save_interval=0.0001)

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

        restore_trainer = Trainer(self.model, self.optimizer,
                                  self.iterator, self.instances, num_epochs=2,
                                  serialization_dir=self.TEST_DIR,
                                  model_save_interval=0.0001)
        epoch, _ = restore_trainer._restore_checkpoint() # pylint: disable=protected-access
        assert epoch == 2
        # One batch per epoch.
        assert restore_trainer._batch_num_total == 2 # pylint: disable=protected-access


class TestSparseClipGrad(AllenNlpTestCase):
    def test_sparse_clip_grad(self):
        # create a sparse embedding layer, then take gradient
        embedding = torch.nn.Embedding(100, 16, sparse=True)
        embedding.zero_grad()
        ids = torch.autograd.Variable((torch.rand(17) * 100).long())
        # Set some of the ids to the same value so that the sparse gradient
        # has repeated indices.  This tests some additional logic.
        ids[:5] = 5
        loss = embedding(ids).sum()
        loss.backward()
        assert is_sparse(embedding.weight.grad)

        # Now try to clip the gradients.
        _ = sparse_clip_norm([embedding.weight], 1.5)
        # Final norm should be 1.5
        grad = embedding.weight.grad.data.coalesce()
        self.assertAlmostEqual(grad._values().norm(2.0), 1.5, places=5) # pylint: disable=protected-access
