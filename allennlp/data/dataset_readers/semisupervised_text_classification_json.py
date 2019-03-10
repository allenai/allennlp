import itertools
import json
import logging
from io import TextIOWrapper
from typing import Dict

import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class UnlabeledData(object):
    """
    Custom class for opening unlabeled data files.
    If the specified filepath is None, this class will
    return an empty list. Otherwise, it will open the file
    in read mode.
    """
    def __init__(self, fpath: str = None):
        self.fpath = fpath

    def __enter__(self):
        if self.fpath:
            self.file = open(cached_path(self.fpath), 'r')
        else:
            self.file = []
        return self.file

    def __exit__(self):
        if self.file:
            self.file.close()


@DatasetReader.register("semisupervised_text_classification_json")
class SemiSupervisedTextClassificationJsonReader(TextClassificationJsonReader):
    """
    Reads tokens and (optionally) their labels from a from text classification dataset.

    This dataset reader inherits from TextClassificationJSONReader, but differs from its parent
    in that it is primed for semisupervised learning. This dataset reader allows for:
        1) Ignoring labels in the training data (e.g. for unsupervised pretraining)
        2) Reading additional unlabeled data from another file
        3) Throttling the training data to a random subsample (according to the numpy seed),
           for analysis of the effect of semisupervised models on different amounts of labeled
           data

    Expects a "tokens" field and a "label" field in JSON format.

    The output of ``read`` is a list of ``Instances`` with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``, if not ignoring labels.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : ``Tokenizer``, optional (default = ``{"tokens": WordTokenizer()}``)
        Tokenizer to split the input text into words or other kinds of tokens.
    segment_sentences: ``bool``, optional (default = ``False``)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences,
        like the Hierarchical Attention Network.
    sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    ignore_labels: ``bool``, optional (default = ``False``)
        If specified, will ignore labels when reading data.
    additional_unlabeled_data_path: ``str``, optional (default = ``None``)
        If specified, will additionally read all unlabeled data from this filepath.
        If ignore_labels is set to False, all data in this file should have a
        consistent dummy-label (e.g. "N/A"), to identify examples that are unlabeled
        in a downstream model that uses this dataset reader.
    sample: ``int``, optional (default = ``None``)
        If specified, will sample data to a specified length.
            **Note**:
                1) This operation will *not* apply to any additional unlabeled data
                   (specified in `additional_unlabeled_data_path`).
                2) To produce a consistent subsample of data, use a consistent seed in your
                   training config.
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 segment_sentences: bool = False,
                 max_sequence_length: int = None,
                 skip_label_indexing: bool = False,
                 ignore_labels: bool = False,
                 additional_unlabeled_data_path: str = None,
                 sample: int = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy,
                         token_indexers=token_indexers,
                         tokenizer=tokenizer,
                         segment_sentences=segment_sentences,
                         max_sequence_length=max_sequence_length,
                         skip_label_indexing=skip_label_indexing,
                         )
        self._tokenizer = tokenizer or WordTokenizer()
        self._sample = sample
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._ignore_labels = ignore_labels
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._additional_unlabeled_data_path = additional_unlabeled_data_path
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    def _reservoir_sampling(self, file_: TextIOWrapper):
        """
        A function for reading random lines from file without loading the 
        entire file into memory. For more information, see here: https://en.wikipedia.org/wiki/Reservoir_sampling


        To create a k-length sample of a file, without knowing the length of the file in advance, 
        we first create a reservoir array containing the first k elements of the file. Then, we further
        iterate through the file, replacing elements in the reservoir with decreasing probability.

        By induction, one can prove that if there are n items in the file, each item is sampled with probability
        k / n.

        Parameters
        ----------
        file : `_io.TextIOWrapper` - file path
        sample_size : `int` - size of random sample you want

        Returns
        -------
        result : `List[str]` - sample lines of file
        """
        # instantiate file iterator
        file_iterator = iter(file_)

        try:
            # fill the reservoir array
            result = [next(file_iterator) for _ in range(self._sample)]
        except StopIteration:
            raise ConfigurationError(f"sample size {self._sample} larger than number of lines in file.")

        # replace elements in reservoir array with decreasing probability
        for index, item in enumerate(file_iterator, start=self._sample):
            sample_index = np.random.randint(0, index)
            if sample_index < self._sample:
                result[sample_index] = item

        for line in result:
            yield line

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file, UnlabeledData(self._additional_unlabeled_data_path) as unlabeled_data_file:
            if self._sample is not None:
                data_file = self._reservoir_sampling(data_file)
            else:
                data_file = data_file
            file_iterator = itertools.chain(data_file, unlabeled_data_file)
            for line in file_iterator:
                items = json.loads(line)
                text = items["text"]
                if self._ignore_labels:
                    instance = self.text_to_instance(text=text, label=None)
                else:
                    label = str(items.get('label'))
                    instance = self.text_to_instance(text=text, label=label)
                if instance is not None and instance.fields['tokens'].tokens:
                    yield instance
