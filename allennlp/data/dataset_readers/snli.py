from typing import Dict
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


@DatasetReader.register("snli")
class SnliReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

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
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as snli_file:
            logger.info("Reading SNLI instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)

                label = example["gold_label"]
                if label == "-":
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the training data.
                    continue

                premise = example["sentence1"]
                hypothesis = example["sentence2"]

                yield self.text_to_instance(premise, hypothesis, label)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        label: str = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}

        if self._combine_input_fields:
            tokens = self._tokenizer.tokenize_sentence_pair(premise, hypothesis)  # type: ignore
            fields["tokens"] = TextField(tokens, self._token_indexers)
        else:
            premise_tokens = self._tokenizer.tokenize(premise)
            hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
            fields["premise"] = TextField(premise_tokens, self._token_indexers)
            fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)

            metadata = {
                "premise_tokens": [x.text for x in premise_tokens],
                "hypothesis_tokens": [x.text for x in hypothesis_tokens],
            }
            fields["metadata"] = MetadataField(metadata)

        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)
