# pylint: disable=invalid-name,no-self-use,abstract-method
from typing import Iterator

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import train_model
from allennlp.data import LazyDataset, Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader

@DatasetReader.register('lazy-sequence-tagger')
class LazySequenceTaggerDatasetReader(DatasetReader):
    """
    This is a dumb hack to get a LazyDataset
    """
    def __init__(self) -> None:
        self.reader = SequenceTaggingDatasetReader()

    def read(self, file_path: str):
        def generator() -> Iterator[Instance]:
            """
            Read from the file every time generator is called
            """
            yield from self.reader.read(file_path).iterinstances()

        return LazyDataset(generator)

    @classmethod
    def from_params(cls, params: Params) -> 'LazySequenceTaggingDatasetReader':
        return cls()


class TestLazyTrain(AllenNlpTestCase):
    def test_train_model(self):
        params = Params({
                "model": {
                        "type": "simple_tagger",
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
                },
                "dataset_reader": {"type": "lazy-sequence-tagger"},
                "train_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "validation_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

        train_model(params, serialization_dir=self.TEST_DIR)

    def test_train_with_test_set(self):
        params = Params({
                "model": {
                        "type": "simple_tagger",
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
                },
                "dataset_reader": {"type": "lazy-sequence-tagger"},
                "train_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "test_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "validation_data_path": 'tests/fixtures/data/sequence_tagging.tsv',
                "evaluate_on_test": True,
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
        })

        train_model(params, serialization_dir=self.TEST_DIR)
