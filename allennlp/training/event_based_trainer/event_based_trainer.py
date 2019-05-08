"""
blah
"""
from typing import Dict, Any, Union, List, Iterable, Callable
from collections import defaultdict

from allennlp.common.tqdm import Tqdm
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models.model import Model
from allennlp.training.event_based_trainer.callback import Callback
from allennlp.training.event_based_trainer.events import Events
from allennlp.training.trainer_base import TrainerBase


class EventBasedTrainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 epochs_generator: Iterable[Iterable[TensorDict]],
                 serialization_dir: str,
                 cuda_device: Union[int, List] = -1) -> None:
        super().__init__(serialization_dir, cuda_device)
        self.model = self._move_to_gpu(model)
        self.epochs_generator = epochs_generator
        self.callbacks: Dict[str, List[Callback]] = defaultdict(list)

        # Keeping track
        self.batch_number = 0
        self.epoch_number = 0
        self.should_terminate = False
        self.tqdm = None

        # Current batch, current output
        self.batch: TensorDict = None
        self.output: Dict[str, Any] = None

        # Any other state a callback might want to use
        self.state: Dict[str, Any] = {}

    def add_callback(self, event: str, callback: Callback) -> None:
        self.callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callback) -> None:
        event_callbacks = self.callbacks.get(event)
        if event_callbacks:
            self.callbacks[event] = [cb for cb in event_callbacks if cb != callback]

    def fire_event(self, event: str) -> None:
        for callback in self.callbacks.get(event, ()):
            callback.call(self)

    def train(self) -> Dict[str, Any]:
        self.batch_number = 0
        self.epoch_number = 0
        train_metrics = []

        self.fire_event(Events.TRAINING_START)
        for epoch in self.epochs_generator:
            self.fire_event(Events.EPOCH_START)

            self.tqdm = Tqdm.tqdm(epoch)
            for batch in self.tqdm:
                # TODO(move to cuda device)

                self.batch = batch
                self.fire_event(Events.BATCH_START)
                self.fire_event(Events.PROCESS_BATCH)
                self.fire_event(Events.BATCH_END)

                if self.should_terminate:
                    break

                self.batch_number += 1

            if self.should_terminate:
                break

            train_metrics.append(self.model.get_metrics(reset=True))

            self.fire_event(Events.EPOCH_END)
            self.epoch_number += 1

        self.fire_event(Events.TRAINING_END)

        return train_metrics[-1]

    def terminate(self) -> None:
        self.should_terminate = True
