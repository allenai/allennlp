import datasets
from allennlp.tango.dataset import AllenNlpDataset
from allennlp.tango.step import Step


@Step.register("hf_dataset")
class HuggingfaceDataset(Step):
    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # These are already cached by huggingface.

    def run(self, dataset_name: str) -> AllenNlpDataset:  # type: ignore
        return AllenNlpDataset(datasets.load_dataset(dataset_name), None, {"source": "huggingface"})
