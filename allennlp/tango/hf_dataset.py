"""
*AllenNLP Tango is an experimental API and parts of it might change or disappear
every time we release a new version.*
"""

import datasets
from allennlp.tango.dataset import DatasetDict
from allennlp.tango.step import Step


@Step.register("hf_dataset")
class HuggingfaceDataset(Step):
    """This steps reads a huggingface dataset and returns it in `DatasetDict` format."""

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # These are already cached by huggingface.

    def run(self, dataset_name: str) -> DatasetDict:  # type: ignore
        """Reads and returns a huggingface dataset. `dataset_name` is the name of the dataset."""
        return DatasetDict(datasets.load_dataset(dataset_name), None, {"source": "huggingface"})
