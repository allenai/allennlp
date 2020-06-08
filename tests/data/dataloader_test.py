import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances


def test_multi_processing_with_lazy_dataset_raises():
    with pytest.raises(ConfigurationError, match=r".*'num_workers'.*"):
        DataLoader(_LazyInstances(iter([]), "nonexistent_file"), num_workers=1)
