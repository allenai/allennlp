from typing import Iterable

import pytest

from allennlp.data.instance import Instance
from allennlp.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_reader import AllennlpLazyDataset


def fake_instance_generator(file_name: str) -> Iterable[Instance]:
    yield from []


def test_multi_processing_with_lazy_dataset_warns():
    with pytest.warns(UserWarning, match=r".*deadlocks.*"):
        DataLoader(AllennlpLazyDataset(fake_instance_generator, "nonexistent_file"), num_workers=1)
