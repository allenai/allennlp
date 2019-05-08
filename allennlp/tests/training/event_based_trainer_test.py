# pylint: disable=invalid-name,too-many-public-methods,protected-access
from typing import Iterable

import torch

from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.tqdm import Tqdm
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.training import Trainer
from allennlp.training.event_based_trainer.callback import Callback
from allennlp.training.event_based_trainer.event_based_trainer import EventBasedTrainer
from allennlp.training.event_based_trainer.events import Events


class TrainOneBatch(Callback):
    """
    Trains one batch by running
        optimizer.zero_grad -> model.forward -> loss.backward -> optimizer.step
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def call(self, trainer: EventBasedTrainer) -> None:
        self.optimizer.zero_grad()

        trainer.output = trainer.model(**trainer.batch)
        loss = trainer.output["loss"]
        loss.backward()
        trainer.tqdm.set_description(f"loss: {loss}")

        self.optimizer.step()


class Evaluate(Callback):
    def __init__(self,
                 epoch_generator: Iterable[TensorDict],
                 early_stopping_metric: str = None,
                 patience: int = None) -> None:
        self.epoch_generator = epoch_generator
        self.early_stopping_metric = early_stopping_metric
        self.patience = patience
        self.metrics_history = []

    def call(self, trainer: EventBasedTrainer) -> None:
        trainer.model.eval()

        for batch in self.epoch_generator:
            trainer.model.forward(**batch)

        metrics = trainer.model.get_metrics(reset=True)
        self.metrics_history.append(metrics)
        trainer.model.train()

        if self.early_stopping_metric:
            # TODO: early_stopping logic would go here
            trainer.terminate()


class MarkDone(Callback):
    """
    Callback that just maintains a "done" flag
    """
    def __init__(self) -> None:
        self.done = False

    def call(self, trainer: EventBasedTrainer) -> None:
        self.done = True




class TestEventBasedTrainer(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
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

    def test_trainer_can_run(self):
        num_epochs = 20

        epochs_generator = (self.iterator(self.instances, num_epochs=1)
                            for _ in range(num_epochs))

        trainer = EventBasedTrainer(model=self.model,
                                    epochs_generator=epochs_generator,
                                    serialization_dir=self.TEST_DIR,
                                    cuda_device=-1)

        process_batch = TrainOneBatch(optimizer=self.optimizer)
        validate = Evaluate(self.iterator(self.instances, num_epochs=1))
        mark_done = MarkDone()
        evaluate_on_test = Evaluate(self.iterator(self.instances, num_epochs=1))

        # trainer.add_callback(Events.TRAINING_START, set_up_logging)
        trainer.add_callback(Events.PROCESS_BATCH, process_batch)
        # trainer.add_callback(Events.BATCH_END, log_to_tensorboard)
        trainer.add_callback(Events.EPOCH_END, validate)
        # trainer.add_callback(Events.EPOCH_END, checkpoint_model)
        trainer.add_callback(Events.TRAINING_END, evaluate_on_test)
        trainer.add_callback(Events.TRAINING_END, mark_done)

        trainer.train()

        assert mark_done.done
        assert trainer.epoch_number == 20
