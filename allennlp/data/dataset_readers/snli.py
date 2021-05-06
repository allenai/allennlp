from typing import Dict, Optional
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)


def maybe_collapse_label(label: str, collapse: bool):
    """
    Helper function that optionally collapses the "contradiction" and "neutral" labels
    into "non-entailment".
    """
    assert label in ["contradiction", "neutral", "entailment"]
    if collapse and label in ["contradiction", "neutral"]:
        return "non-entailment"
    return label


@DatasetReader.register("snli")
class SnliReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.
    Registered as a `DatasetReader` with name "snli".
    # Parameters
    tokenizer : `Tokenizer`, optional (default=`SpacyTokenizer()`)
        We use this `Tokenizer` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    combine_input_fields : `bool`, optional
            (default=`isinstance(tokenizer, PretrainedTransformerTokenizer)`)
        If False, represent the premise and the hypothesis as separate fields in the instance.
        If True, tokenize them together using `tokenizer.tokenize_sentence_pair()`
        and provide a single `tokens` field in the instance.
    collapse_labels : `bool`, optional (default=`False`)
        If `True`, the "neutral" and "contradiction" labels will be collapsed into "non-entailment";
        "entailment" will be left unchanged.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        collapse_labels: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._tokenizer = tokenizer or SpacyTokenizer()
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)
        self.collapse_labels = collapse_labels

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r") as snli_file:
            example_iter = (json.loads(line) for line in snli_file)
            filtered_example_iter = (
                example for example in example_iter if example.get("gold_label") != "-"
            )
            for example in self.shard_iterable(filtered_example_iter):
                label = example.get("gold_label")
                premise = example["sentence1"]
                hypothesis = example["sentence2"]
                yield self.text_to_instance(premise, hypothesis, label)

    @overrides
    def text_to_instance(self, premise, hypothesis, label: str = None) -> Instance:  # type: ignore

        fields: Dict[str, Field] = {}
        premise = self._tokenizer.tokenize(premise)
        hypothesis = self._tokenizer.tokenize(hypothesis)

        if self._combine_input_fields:
            tokens = self._tokenizer.add_special_tokens(premise, hypothesis)
            fields["tokens"] = TextField(tokens)
        else:
            premise_tokens = self._tokenizer.add_special_tokens(premise)
            hypothesis_tokens = self._tokenizer.add_special_tokens(hypothesis)
            fields["premise"] = TextField(premise_tokens)
            fields["hypothesis"] = TextField(hypothesis_tokens)

            metadata = {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            }
            fields["metadata"] = MetadataField(metadata)

        if label:
            maybe_collapsed_label = maybe_collapse_label(label, self.collapse_labels)
            fields["label"] = LabelField(maybe_collapsed_label)

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance):
        if "tokens" in instance.fields:
            instance.fields["tokens"]._token_indexers = self._token_indexers  # type: ignore
        else:
            instance.fields["premise"]._token_indexers = self._token_indexers  # type: ignore
            instance.fields["hypothesis"]._token_indexers = self._token_indexers  # type: ignore
