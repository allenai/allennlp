import json
from typing import List

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField
from allennlp.common import Params

import numpy as np

@DatasetReader.register("array")
class ArrayReader(DatasetReader):
    def read(self, file_path: str) -> Dataset:
        """
        Actually reads some data from the `file_path` and returns a :class:`Dataset`.
        """
        instances = []  # type: List[Instance]
        with open(file_path) as input_file:
            for line in input_file:
                blob = json.loads(line)
                fields = {key: ArrayField(np.array(value, dtype=np.float32)) for key, value in blob.items()}
                instance = Instance(fields)
                instances.append(instance)

        return Dataset(instances)


    @classmethod
    def from_params(cls, params: Params) -> 'ArrayReader':
        # choice = params.pop_choice('type', cls.list_available())
        return ArrayReader()
