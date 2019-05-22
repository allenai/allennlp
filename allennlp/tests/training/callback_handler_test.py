# pylint: disable=invalid-name,too-many-public-methods,protected-access
from typing import Iterable, Callable, Optional, List
from typing_extensions import Protocol

import torch

from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models.model import Model
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.training.callbacks.callback import Callback
from allennlp.training.callbacks.callback_handler import CallbackHandler
from allennlp.training.callbacks.events import Events


class ValidateState(Protocol):
    validation_metrics: List[dict]
    model: Model
    early_stopping_metric: Optional[str]
    terminate: bool


class Validate(Callback[ValidateState]):
    def __init__(self,
                 epoch_generator: Callable[[], Iterable[TensorDict]],
                 trigger_event: str = Events.EPOCH_END,
                 early_stopping_metric: str = None,
                 patience: int = None) -> None:
        self.epoch_generator = epoch_generator
        self.early_stopping_metric = early_stopping_metric
        self.patience = patience
        self.trigger_event = trigger_event

    def __call__(self, event: str, state: ValidateState) -> None:
        if event == Events.TRAINING_START:
            state.validation_metrics = []

        if event == self.trigger_event:
            state.model.eval()

            loss = 0.0

            for batch in self.epoch_generator():
                output = state.model.forward(**batch)
                loss += output["loss"]

            metrics = state.model.get_metrics(reset=True)
            metrics["loss"] = torch.sum(loss).item()
            state.validation_metrics.append(metrics)
            state.model.train()

            if self.early_stopping_metric:
                # TODO: early_stopping logic would go here
                state.terminate = True


class MarkDoneState(Protocol):
    done: bool

class MarkDone(Callback[MarkDoneState]):
    """
    Callback that just sets a done 'flag'
    """
    def __init__(self, trigger_event: str = Events.TRAINING_END) -> None:
        self.trigger_event = trigger_event

    def __call__(self, event: str, state: MarkDoneState) -> None:
        if event == self.trigger_event:
            state.done = True


class State:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.validation_metrics: List[dict] = None
        self.early_stopping_metric: Optional[str] = None
        self.terminate: bool = False
        self.done: bool = False

class TestCallbackHandler(AllenNlpTestCase):
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


        callbacks = [
                Validate(lambda: self.iterator(self.instances, num_epochs=1)),
                MarkDone()
        ]

        state = State(model=self.model)
        handler = CallbackHandler(callbacks, state)

        handler.fire_event(Events.TRAINING_START)
        for _ in range(num_epochs):
            handler.fire_event(Events.EPOCH_START)
            self.model.train()
            for batch in self.iterator(self.instances, num_epochs=1):

                self.optimizer.zero_grad()

                handler.fire_event(Events.BATCH_START)

                output = self.model(**batch)

                handler.fire_event(Events.AFTER_FORWARD)

                output["loss"].backward()

                handler.fire_event(Events.AFTER_BACKWARD)

                self.optimizer.step()

                handler.fire_event(Events.BATCH_END)

                if state.terminate:
                    break

            self.model.eval()

            handler.fire_event(Events.EPOCH_END)

            if state.terminate:
                break

        handler.fire_event(Events.TRAINING_END)

        assert state.done
        assert len(state.validation_metrics) == 20
