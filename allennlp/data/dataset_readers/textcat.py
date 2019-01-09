from typing import Dict, List
import logging
import json
import numpy as np
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("textcat")
class TextCatReader(DatasetReader):
    """
    Reads tokens and their labels from a labeled text classification dataset.
    Expects a "tokens" field and a "category" field in JSON format.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    word_tokenizer : ``Tokenizer``, optional (default = ``{"tokens": WordTokenizer()}``)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    segment_sentences: ``bool``, optional (default = ``False``)
        If true, we will first segment the text into sentences using Spacy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences,
        like the Hierarchical Attention Network.
    sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    debug : ``bool``, optional (default = ``False``)
        If true, will only read 100 instances from file(s), so data can be read quickly during debugging.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_tokenizer: Tokenizer = None,
                 segment_sentences: bool = False,
                 sequence_length: int = None,
                 debug: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.debug = debug
        self._word_tokenizer = word_tokenizer or WordTokenizer()
        self._segment_sentences = segment_sentences
        self._sequence_length = sequence_length
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            if self.debug:
                lines = np.random.choice(data_file.readlines(), 100)
            else:
                lines = data_file.readlines()
        for line in lines:
            if not line:
                continue
            items = json.loads(line)
            tokens = items["tokens"]
            label = str(items["label"])
            instance = self.text_to_instance(tokens=tokens,
                                             label=label)
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         label: str = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't
        have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given text.
        label ``str``, optional, (default = None).
            The label for this text.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_fields = []
        if self._segment_sentences:
            sentence_tokens = self._sentence_segmenter.split_sentences(tokens)
            if not sentence_tokens:
                return None
            for sentence in sentence_tokens:
                word_tokens = self._word_tokenizer.tokenize(sentence)
                if self._sequence_length is not None:
                    if len(word_tokens) > self._sequence_length:
                        word_tokens = word_tokens[:self._sequence_length]
                    else:
                        padding = [Token("@@PADDING")] * (self._sequence_length - len(word_tokens))
                        word_tokens = word_tokens + padding
                text_fields.append(TextField(word_tokens, self._token_indexers))
            fields['tokens'] = ListField(text_fields)
        else:
            tokens_ = self._word_tokenizer.tokenize(tokens)
            if not tokens_:
                return None
            if self._sequence_length is not None:
                if len(tokens_) > self._sequence_length:
                    tokens_ = tokens_[:self._sequence_length]
                else:
                    padding = [Token("@@PADDING")] * (self._sequence_length - len(tokens_))
                    tokens_ = tokens_ + padding
            fields['tokens'] = TextField(tokens_,
                                         self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
