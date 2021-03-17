from typing import Dict

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer


@DatasetReader.register("t5")
class T5DatasetReader(DatasetReader):
    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self.tokenizer = PretrainedTransformerTokenizer(model_name)
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(model_name)}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path)) as data_file:
            for line in self.shard_iterable(data_file):
                source, target = line.strip().split("\t")
                yield self.text_to_instance(source, target)

    @overrides
    def text_to_instance(self, source: str, target: str = None) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}
        source_tokens = self.tokenizer.tokenize(source)
        fields["source_tokens"] = TextField(source_tokens)
        if target is not None:
            target_tokens = self.tokenizer.tokenize(target)
            fields["target_tokens"] = TextField(target_tokens)
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self.token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self.token_indexers  # type: ignore
