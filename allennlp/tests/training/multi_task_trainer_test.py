"""
This isn't testing a real class, it's a proof-of-concept
for how multi-task training could work. This is certainly
not the only way to do multi-task training using AllenNLP.

Note that you could almost fit this whole setup into
the "SingleTaskTrainer" paradigm, if you just wrote like a
``MinglingDatasetReader`` that wrapped multiple dataset readers.
The main problem is that the ``SingleTaskTrainer`` expects
a single ``train_path``. (Even that you could fudge by passing
in a Dict[str, str] serialized as JSON, but that's really hacky.)
"""
# pylint: disable=bad-continuation

from typing import List, Dict, Iterable, Any, Set
from collections import defaultdict
import os

import tqdm
import torch

from allennlp.common import Registrable
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.iterators import DataIterator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase

@DatasetReader.register("multi-task-test")
class MyReader(DatasetReader):
    """
    Just reads in a text file and sticks each line
    in a ``TextField`` with the specified name.
    """
    def __init__(self, field_name: str) -> None:
        super().__init__()
        self.field_name = field_name
        self.tokenizer = WordTokenizer()
        self.token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokens = self.tokenizer.tokenize(sentence)
        return Instance({self.field_name: TextField(tokens, self.token_indexers)})

    def _read(self, file_path: str):
        with open(file_path) as data_file:
            for line in data_file:
                yield self.text_to_instance(line)


class DatasetMingler(Registrable):
    """
    Our ``DataIterator`` class expects a single dataset;
    this is an abstract class for combining multiple datasets into one.

    You could imagine an alternate design where there is a
    ``MinglingDatasetReader`` that wraps multiple dataset readers,
    but then somehow you'd have to get it multiple file paths.
    """
    def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        raise NotImplementedError


@DatasetMingler.register("round-robin")
class RoundRobinMingler(DatasetMingler):
    """
    Cycle through datasets, ``take_at_time`` instances at a time.
    """
    def __init__(self,
                 dataset_name_field: str = "dataset",
                 take_at_a_time: int = 1) -> None:
        self.dataset_name_field = dataset_name_field
        self.take_at_a_time = take_at_a_time

    def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
        iterators = {name: iter(dataset) for name, dataset in datasets.items()}
        done: Set[str] = set()

        while iterators.keys() != done:
            for name, iterator in iterators.items():
                if name not in done:
                    try:
                        for _ in range(self.take_at_a_time):
                            instance = next(iterator)
                            instance.fields[self.dataset_name_field] = MetadataField(name)
                            yield instance
                    except StopIteration:
                        done.add(name)


@DataIterator.register("homogeneous-batch")
class HomogeneousBatchIterator(DataIterator):
    """
    An iterator that takes instances of various types
    and yields single-type batches of them. There's a flag
    to allow mixed-type batches, but at that point you might
    as well just use ``BasicIterator``?
    """
    def __init__(self,
                 type_field_name: str = "dataset",
                 allow_mixed_batches: bool = False,
                 batch_size: int = 32) -> None:
        super().__init__(batch_size)
        self.type_field_name = type_field_name
        self.allow_mixed_batches = allow_mixed_batches

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        """
        This method should return one epoch worth of batches.
        """
        hoppers: Dict[Any, List[Instance]] = defaultdict(list)

        for instance in instances:
            # Which hopper do we put this instance in?
            if self.allow_mixed_batches:
                instance_type = ""
            else:
                instance_type = instance.fields[self.type_field_name].metadata  # type: ignore

            hoppers[instance_type].append(instance)

            # If the hopper is full, yield up the batch and clear it.
            if len(hoppers[instance_type]) >= self._batch_size:
                yield Batch(hoppers[instance_type])
                hoppers[instance_type].clear()

        # Deal with leftovers
        for remaining in hoppers.values():
            if remaining:
                yield Batch(remaining)


@Model.register("multi-task-test")
class MyModel(Model):
    """
    This model does nothing interesting, but it's designed to
    operate on heterogeneous instances using shared parameters
    (well, one shared parameter) like you'd have in multi-task training.
    """
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.weight = torch.nn.Parameter(torch.randn(()))

    def forward(self,  # type: ignore
                dataset: List[str],
                field_a: torch.Tensor = None,
                field_b: torch.Tensor = None,
                field_c: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        loss = torch.tensor(0.0)   # pylint: disable=not-callable
        if dataset[0] == "a":
            loss += field_a["tokens"].sum() * self.weight
        elif dataset[0] == "b":
            loss -= field_b["tokens"].sum() * self.weight ** 2
        elif dataset[0] == "c":
            loss += field_c["tokens"].sum() * self.weight ** 3
        else:
            raise ValueError(f"unknown dataset: {dataset[0]}")

        return {"loss": loss}


@TrainerBase.register("multi-task-test")
class MultiTaskTrainer(TrainerBase):
    """
    A simple trainer that works in our multi-task setup.
    Really the main thing that makes this task not fit into our
    existing trainer is the multiple datasets.
    """
    def __init__(self,
                 model: Model,
                 serialization_dir: str,
                 iterator: DataIterator,
                 mingler: DatasetMingler,
                 optimizer: torch.optim.Optimizer,
                 datasets: Dict[str, Iterable[Instance]],
                 num_epochs: int = 10,
                 num_serialized_models_to_keep: int = 10) -> None:
        super().__init__(serialization_dir)
        self.model = model
        self.iterator = iterator
        self.mingler = mingler
        self.optimizer = optimizer
        self.datasets = datasets
        self.num_epochs = num_epochs
        self.checkpointer = Checkpointer(serialization_dir,
                                         num_serialized_models_to_keep=num_serialized_models_to_keep)

    def save_checkpoint(self, epoch: int) -> None:
        training_state = {"epoch": epoch, "optimizer": self.optimizer.state_dict()}
        self.checkpointer.save_checkpoint(epoch, self.model.state_dict(), training_state, True)

    def restore_checkpoint(self) -> int:
        model_state, trainer_state = self.checkpointer.restore_checkpoint()
        if not model_state and not trainer_state:
            return 0
        else:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(trainer_state["optimizer"])
            return trainer_state["epoch"] + 1


    def train(self) -> Dict:
        start_epoch = self.restore_checkpoint()

        self.model.train()
        for epoch in range(start_epoch, self.num_epochs):
            total_loss = 0.0
            batches = tqdm.tqdm(self.iterator(self.mingler.mingle(self.datasets), num_epochs=1))
            for i, batch in enumerate(batches):
                self.optimizer.zero_grad()
                loss = self.model.forward(**batch)['loss']
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
                batches.set_description(f"epoch: {epoch} loss: {total_loss / (i + 1)}")

            # Save checkpoint
            self.save_checkpoint(epoch)

        return {}

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None) -> 'MultiTaskTrainer':
        readers = {name: DatasetReader.from_params(reader_params)
                   for name, reader_params in params.pop("train_dataset_readers").items()}
        train_file_paths = params.pop("train_file_paths").as_dict()

        datasets = {name: reader.read(train_file_paths[name])
                    for name, reader in readers.items()}

        instances = (instance for dataset in datasets.values() for instance in dataset)
        vocab = Vocabulary.from_params(Params({}), instances)
        model = Model.from_params(params.pop('model'), vocab=vocab)
        iterator = DataIterator.from_params(params.pop('iterator'))
        iterator.index_with(vocab)
        mingler = DatasetMingler.from_params(params.pop('mingler'))

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop('optimizer'))

        num_epochs = params.pop_int("num_epochs", 10)

        _ = params.pop("trainer", Params({}))

        params.assert_empty(__name__)

        return MultiTaskTrainer(model, serialization_dir, iterator, mingler, optimizer, datasets, num_epochs)



class MultiTaskTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        params = Params({
            "model": {
                "type": "multi-task-test"
            },
            "iterator": {
                "type": "homogeneous-batch"
            },
            "mingler": {
                "type": "round-robin"
            },
            "optimizer": {
                "type": "sgd",
                "lr": 0.01
            },
            "train_dataset_readers": {
                "a": {
                    "type": "multi-task-test",
                    "field_name": "field_a"
                },
                "b": {
                    "type": "multi-task-test",
                    "field_name": "field_b"
                },
                "c": {
                    "type": "multi-task-test",
                    "field_name": "field_c"
                },
            },
            "train_file_paths": {
                "a": self.FIXTURES_ROOT / 'data' / 'babi.txt',
                "b": self.FIXTURES_ROOT / 'data' / 'conll2000.txt',
                "c": self.FIXTURES_ROOT / 'data' / 'conll2003.txt'
            },
            "trainer": {
                "type": "multi-task-test"
            }
        })

        self.trainer = TrainerBase.from_params(params, self.TEST_DIR)

    def test_training(self):
        self.trainer.train()

        assert os.path.exists(os.path.join(self.TEST_DIR, "best.th"))
