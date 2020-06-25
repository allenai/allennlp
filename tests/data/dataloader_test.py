from typing import Iterable

import pytest

from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
from allennlp.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_reader import (
    DatasetReader,
    AllennlpLazyDataset,
)


def test_multi_processing_with_lazy_dataset_warns():
    def fake_instance_generator(file_name: str) -> Iterable[Instance]:
        yield from []

    with pytest.warns(UserWarning, match=r".*deadlocks.*"):
        DataLoader(AllennlpLazyDataset(fake_instance_generator, "nonexistent_file"), num_workers=1)


@pytest.mark.parametrize("lazy", (True, False))
def test_loader_uses_all_instances_when_batches_per_epochs_set(lazy):
    NUM_INSTANCES = 20
    BATCH_SIZE = 2
    BATCHES_PER_EPOCH = 3
    EPOCHS = 4

    class FakeDatasetReader(DatasetReader):
        def _read(self, filename: str) -> Iterable[Instance]:
            for i in range(NUM_INSTANCES):
                yield Instance({"index": LabelField(i, skip_indexing=True)})

    reader = FakeDatasetReader(lazy=lazy)
    dataset = reader.read("blah")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, batches_per_epoch=BATCHES_PER_EPOCH)
    epoch_batches = []
    for epoch in range(EPOCHS):
        batches = []
        for batch in loader:
            instances = []
            for index in batch["index"]:
                instances.append(index)
            batches.append(instances)
        epoch_batches.append(batches)

    assert epoch_batches == [
        # Epoch 0.
        [[0, 1], [2, 3], [4, 5]],
        # Epoch 1.
        [[6, 7], [8, 9], [10, 11]],
        # Epoch 2.
        [[12, 13], [14, 15], [16, 17]],
        # Epoch 3.
        [[18, 19], [0, 1], [2, 3]],
    ]
