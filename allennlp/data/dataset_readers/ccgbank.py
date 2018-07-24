from typing import Dict, List
import logging
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ccgbank")
class CcgBankDatasetReader(DatasetReader):
    """
    Reads data in the "machine-readable derivation" format of the CCGbank dataset.
    (see https://catalog.ldc.upenn.edu/docs/LDC2005T13/CCGbankManual.pdf, section D.2)

    In particular, it pulls out the leaf nodes, which are represented as

        (<L ccg_category modified_pos original_pos token predicate_arg_category>)

    The tarballed version of the dataset contains many files worth of this data,
    in files /data/AUTO/xx/wsj_xxxx.auto

    This dataset reader expects a single text file. Accordingly, if you're using that dataset,
    you'll need to first concatenate some of those files into a training set, a validation set,
    and a test set.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading instances from lines in file at: %s", file_path)

        with open(file_path) as input_file:
            for line in input_file:
                if line.startswith("(<"):
                    # Each leaf looks like
                    # (<L ccg_category modified_pos original_pos token predicate_arg_category>)
                    leaves = re.findall("<L (.*?)>", line)

                    # Use magic unzipping trick to split into tuples
                    tuples = zip(*[leaf.split() for leaf in leaves])

                    # Convert to lists and assign to variables.
                    ccg_categories, modified_pos_tags, original_pos_tags, tokens, predicate_arg_categories = \
                            [list(result) for result in tuples]

                    yield self.text_to_instance(tokens,
                                                ccg_categories,
                                                modified_pos_tags,
                                                original_pos_tags,
                                                predicate_arg_categories)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         ccg_categories: List[str] = None,
                         original_pos_tags: List[str] = None,
                         modified_pos_tags: List[str] = None,
                         predicate_arg_categories: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        ccg_categories : ``List[str]``, optional, (default = None).
            The CCG categories for the words in the sentence. (e.g. N/N)
        original_pos_tags : ``List[str]``, optional, (default = None).
            The tag assigned to the word in the Penn Treebank.
        modified_pos_tags : ``List[str]``, optional, (default = None).
            The POS tag might have changed during the translation to CCG.
        predicate_arg_categories : ``List[str]``, optional, (default = None).
            Encodes the word-word dependencies in the underlying predicate-
            argument structure.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence.
            ccg_categories : ``SequenceLabelField``
                The CCG categories (only if supplied)
            original_pos_tags : ``SequenceLabelField``
                Original POS tag (only if supplied)
            modified_pos_tags : ``SequenceLabelField``
                Modified POS tag (only if supplied)
            predicate_arg_categories : ``SequenceLabelField``
                Predicate-argument categories (only if supplied)
        """
        # pylint: disable=arguments-differ
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        for field_name, labels in (('ccg_categories', ccg_categories),
                                   ('original_pos_tags', original_pos_tags),
                                   ('modified_pos_tags', modified_pos_tags),
                                   ('predicate_arg_categories', predicate_arg_categories)):
            if labels is not None:
                fields[field_name] = SequenceLabelField(labels, text_field)

        return Instance(fields)
