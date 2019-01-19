# pylint: disable=invalid-name,too-many-public-methods
from typing import Dict

import torch
import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.ema_trainer import EMATrainer
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.iterators import BasicIterator
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.models.model import Model


class TestTrainer(AllenNlpTestCase):
    def setUp(self):
        super(TestTrainer, self).setUp()
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
        self.optimizer = torch.optim.SGD(self.model.parameters(), 0.01)
        self.iterator = BasicIterator(batch_size=2)
        self.iterator.index_with(vocab)

    def test_trainer_can_run(self):
        trainer = EMATrainer(model=self.model,
                             optimizer=self.optimizer,
                             iterator=self.iterator,
                             train_dataset=self.instances,
                             validation_dataset=self.instances,
                             num_epochs=2,
                             exponential_moving_average_decay=0.9999)
        metrics = trainer.train()
        assert 'best_validation_loss' in metrics
        assert isinstance(metrics['best_validation_loss'], float)
        assert 'best_validation_accuracy' in metrics
        assert isinstance(metrics['best_validation_accuracy'], float)
        assert 'best_validation_accuracy3' in metrics
        assert isinstance(metrics['best_validation_accuracy3'], float)
        assert 'best_epoch' in metrics
        assert isinstance(metrics['best_epoch'], int)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    def test_trainer_can_run_cuda(self):
        trainer = EMATrainer(self.model, self.optimizer,
                             self.iterator, self.instances, num_epochs=2,
                             cuda_device=0,
                             exponential_moving_average_decay=0.9999)
        trainer.train()

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason="Need multiple GPUs.")
    def test_trainer_can_run_multiple_gpu(self):

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
        trainer = EMATrainer(MetaDataCheckWrapper(self.model), self.optimizer,
                             multigpu_iterator, self.instances, num_epochs=2,
                             cuda_device=[0, 1],
                             exponential_moving_average_decay=0.9999)
        metrics = trainer.train()
        assert 'peak_cpu_memory_MB' in metrics
        assert isinstance(metrics['peak_cpu_memory_MB'], float)
        assert metrics['peak_cpu_memory_MB'] > 0
        assert 'peak_gpu_0_memory_MB' in metrics
        assert isinstance(metrics['peak_gpu_0_memory_MB'], int)
        assert 'peak_gpu_1_memory_MB' in metrics
        assert isinstance(metrics['peak_gpu_1_memory_MB'], int)

    def test_trainer_can_resume_training(self):
        trainer = EMATrainer(self.model, self.optimizer,
                             self.iterator, self.instances,
                             validation_dataset=self.instances,
                             num_epochs=1, serialization_dir=self.TEST_DIR)
        trainer.train()
        new_trainer = EMATrainer(self.model, self.optimizer,
                                 self.iterator, self.instances,
                                 validation_dataset=self.instances,
                                 num_epochs=3, serialization_dir=self.TEST_DIR)

        epoch, val_metrics_per_epoch = new_trainer._restore_checkpoint()  # pylint: disable=protected-access
        assert epoch == 1
        assert len(val_metrics_per_epoch) == 1
        assert isinstance(val_metrics_per_epoch[0], float)
        assert val_metrics_per_epoch[0] != 0.
        new_trainer.train()