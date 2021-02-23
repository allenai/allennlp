from typing import Dict, List, Union
import logging
import json
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

logger = logging.getLogger(__name__)


@DatasetReader.register("text_classification_json")
class TextClassificationJsonReader(DatasetReader):
    """
    Reads tokens and their labels from a labeled text classification dataset.

    The output of `read` is a list of `Instance` s with the fields:
        tokens : `TextField` and
        label : `LabelField`

    Registered as a `DatasetReader` with name "text_classification_json".

    [0]: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional
        optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : `Tokenizer`, optional (default = `{"tokens": SpacyTokenizer()}`)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    segment_sentences : `bool`, optional (default = `False`)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences, like [the Hierarchical
        Attention Network][0].
    max_sequence_length : `int`, optional (default = `None`)
        If specified, will truncate tokens to specified maximum length.
    skip_label_indexing : `bool`, optional (default = `False`)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    text_key: `str`, optional (default=`"text"`)
        The key name of the source field in the JSON data file.
    label_key: `str`, optional (default=`"label"`)
        The key name of the target field in the JSON data file.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        segment_sentences: bool = False,
        max_sequence_length: int = None,
        skip_label_indexing: bool = False,
        text_key: str = "text",
        label_key: str = "label",
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._text_key = text_key
        self._label_key = label_key
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in self.shard_iterable(data_file.readlines()):
                if not line:
                    continue
                items = json.loads(line)
                text = items[self._text_key]
                label = items.get(self._label_key)
                if label is not None:
                    if self._skip_label_indexing:
                        try:
                            label = int(label)
                        except ValueError:
                            raise ValueError(
                                "Labels must be integers if skip_label_indexing is True."
                            )
                    else:
                        label = str(label)
                yield self.text_to_instance(text=text, label=label)

    def _truncate(self, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[: self._max_sequence_length]
        return tokens

    @overrides
    def text_to_instance(
        self, text: str, label: Union[str, int] = None
    ) -> Instance:  # type: ignore
        """
        # Parameters

        text : `str`, required.
            The text to classify
        label : `str`, optional, (default = `None`).
            The label for this text.

        # Returns

        An `Instance` containing the following fields:
            - tokens (`TextField`) :
              The tokens in the sentence or phrase.
            - label (`LabelField`) :
              The label label of the sentence or phrase.
        """

        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens))
            fields["tokens"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields["tokens"] = TextField(tokens)
        if label is not None:
            fields["label"] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        if self._segment_sentences:
            for text_field in instance.fields["tokens"]:  # type: ignore
                text_field._token_indexers = self._token_indexers
        else:
            instance.fields["tokens"]._token_indexers = self._token_indexers  # type: ignore
