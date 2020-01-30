from typing import Dict, List, cast, Tuple
import json
import logging
from overrides import overrides

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, BatchSampler, SequentialSampler, SubsetRandomSampler

from allennlp.common.registrable import Registrable
from allennlp.common.util import add_noise_to_dict_values, lazy_groups_of
from allennlp.data.batch import Batch as AllennlpBatch
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Token
from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

logger = logging.getLogger(__name__)

class Dataset(TorchDataset, Registrable):
    def __init__(self):
        self.vocab: Vocabulary = None

    def text_to_instance(self, *inputs) -> Instance:

        raise NotImplementedError

    def __getitem__(self) -> Instance:

        raise NotImplementedError

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab


"""
Here we have two SNLI readers in both of the different styles.
They are only slightly different.
"""


class SnliDataset(Dataset):
    def __init__(
        self, file_path: str, token_indexers: Dict[str, TokenIndexer] = None, lazy: bool = False
    ) -> None:

        super().__init__()

        self._tokenizer = lambda x: [Token(t) for t in x.split(" ")]
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        file_path = cached_path(file_path)
        self.examples = []

        for line in open(file_path, "r"):
            example = json.loads(line)
            if example["gold_label"] == "-":
                # These were cases where the annotators disagreed; we'll just skip them.  It's
                # like 800 out of 500k examples in the training data.
                continue
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Instance:
        example = self.examples[idx]
        instance = self.text_to_instance(
            example["sentence1"], example["sentence2"], example["gold_label"]
        )

        # This is not ideal, we don't want a user to have to worry about this
        # but at the same time, it's expensive and it would be nice if it could happen here.
        # It's possible we could have this in the super class.
        if self.vocab is not None:
            instance.index_fields(self.vocab)
        return instance

    @overrides
    def text_to_instance(self, premise: str, hypothesis: str, label: str = None) -> Instance:

        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer(premise)
        hypothesis_tokens = self._tokenizer(hypothesis)
        fields["premise"] = TextField(premise_tokens, self._token_indexers)
        fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields["label"] = LabelField(label)

        metadata = {
            "premise_tokens": [x.text for x in premise_tokens],
            "hypothesis_tokens": [x.text for x in hypothesis_tokens],
        }
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)


class BatchInstanceSampler(BatchSampler):
    
    def __init__(self, data, batch_size: int, sorting_keys: List[Tuple[str, str]] = None, padding_noise: float = 0.1):

        self.vocab = data.vocab
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._batch_size = batch_size
        self.data = data

    def _argsort_by_padding(self, instances: List[Instance]) -> List[int]:
        """
        Sorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided).  `sorting_keys` is a list of
        `(field_name, padding_key)` tuples.
        """
        if not self._sorting_keys:
            logger.info("No sorting keys given; trying to guess a good one")
            self._guess_sorting_keys(instances)
            logger.info(f"Using {self._sorting_keys} as the sorting keys")
        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            instance.index_fields(self.vocab)
            padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
            if self._padding_noise > 0.0:
                noisy_lengths = {}
                for field_name, field_lengths in padding_lengths.items():
                    noisy_lengths[field_name] = add_noise_to_dict_values(
                        field_lengths, self._padding_noise
                    )
                padding_lengths = noisy_lengths
            instance_with_lengths = (
                [
                    padding_lengths[field_name][padding_key]
                    for (field_name, padding_key) in self._sorting_keys
                ],
                instance,
            )
            instances_with_lengths.append(instance_with_lengths)
        with_indices = [(x, i) for i, x in enumerate(instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return [instance_with_index[-1] for instance_with_index in with_indices]

    def __iter__(self):

        indices = self._argsort_by_padding(self.data)

        for group in lazy_groups_of(indices, self._batch_size):

            yield list(group)

    def _guess_sorting_keys(self, instances: List[Instance]) -> None:
        max_length = 0.0
        longest_padding_key: Tuple[str, str] = None
        for instance in instances:
            instance.index_fields(self.vocab)
            padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
            for field_name, field_padding in padding_lengths.items():
                for padding_key, length in field_padding.items():
                    if length > max_length:
                        max_length = length
                        longest_padding_key = (field_name, padding_key)
        if not longest_padding_key:
            # This shouldn't ever happen (you basically have to have an empty instance list), but
            # just in case...
            raise AssertionError(
                "Found no field that needed padding; we are surprised you got this error, please "
                "open an issue on github"
            )
        self._sorting_keys = [longest_padding_key]


data = SnliDataset("snli_20.jsonl")
vocab = Vocabulary.from_instances(data)
data.index_with(vocab)


sampler = SequentialSampler(data)
batch_sampler = BatchInstanceSampler(data, 4)


def allennlp_collocate(batch):

    batch = AllennlpBatch(batch)
    return batch.as_tensor_dict(batch.get_padding_lengths())

batch_generator = DataLoader(data, batch_sampler=batch_sampler, collate_fn=allennlp_collocate)

iterator = iter(batch_generator)

print()
for i, x in enumerate(batch_generator):

    print(x)
