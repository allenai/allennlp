from typing import Dict, List, Sequence
import logging
import re

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)

_VALID_LABELS = {"ccg", "modified_pos", "original_pos", "predicate_arg"}


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

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    tag_label : `str`, optional (default=`ccg`)
        Specify `ccg`, `modified_pos`, `original_pos`, or `predicate_arg` to
        have that tag loaded into the instance field `tag`.
    feature_labels : `Sequence[str]`, optional (default=`()`)
        These labels will be loaded as features into the corresponding instance fields:
        `ccg` -> `ccg_tags`, `modified_pos` -> `modified_pos_tags`,
        `original_pos` -> `original_pos_tags`, or `predicate_arg` -> `predicate_arg_tags`
        Each will have its own namespace : `ccg_tags`, `modified_pos_tags`,
        `original_pos_tags`, `predicate_arg_tags`. If you want to use one of the tags
        as a feature in your model, it should be specified here.
    label_namespace : `str`, optional (default=`labels`)
        Specifies the namespace for the chosen `tag_label`.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tag_label: str = "ccg",
        feature_labels: Sequence[str] = (),
        label_namespace: str = "labels",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tag_label = tag_label
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))

        self.feature_labels = set(feature_labels)
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))

        self.label_namespace = label_namespace

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
                    (
                        ccg_categories,
                        modified_pos_tags,
                        original_pos_tags,
                        tokens,
                        predicate_arg_categories,
                    ) = [list(result) for result in tuples]

                    yield self.text_to_instance(
                        tokens,
                        ccg_categories,
                        original_pos_tags,
                        modified_pos_tags,
                        predicate_arg_categories,
                    )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokens: List[str],
        ccg_categories: List[str] = None,
        original_pos_tags: List[str] = None,
        modified_pos_tags: List[str] = None,
        predicate_arg_categories: List[str] = None,
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        # Parameters

        tokens : `List[str]`, required.
            The tokens in a given sentence.
        ccg_categories : `List[str]`, optional, (default = None).
            The CCG categories for the words in the sentence. (e.g. N/N)
        original_pos_tags : `List[str]`, optional, (default = None).
            The tag assigned to the word in the Penn Treebank.
        modified_pos_tags : `List[str]`, optional, (default = None).
            The POS tag might have changed during the translation to CCG.
        predicate_arg_categories : `List[str]`, optional, (default = None).
            Encodes the word-word dependencies in the underlying predicate-
            argument structure.

        # Returns

        An `Instance` containing the following fields:
            tokens : `TextField`
                The tokens in the sentence.
            tags : `SequenceLabelField`
                The tags corresponding to the `tag_label` constructor argument.
            feature_label_tags : `SequenceLabelField`
                Tags corresponding to each feature_label (if any) specified in the
                `feature_labels` constructor argument.
        """

        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        # Add "feature labels" to instance
        if "ccg" in self.feature_labels:
            if ccg_categories is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use CCG categories as "
                    "features. Pass them to text_to_instance."
                )
            fields["ccg_tags"] = SequenceLabelField(ccg_categories, text_field, "ccg_tags")
        if "original_pos" in self.feature_labels:
            if original_pos_tags is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use original POS tags as "
                    "features. Pass them to text_to_instance."
                )
            fields["original_pos_tags"] = SequenceLabelField(
                original_pos_tags, text_field, "original_pos_tags"
            )
        if "modified_pos" in self.feature_labels:
            if modified_pos_tags is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use modified POS tags as "
                    " features. Pass them to text_to_instance."
                )
            fields["modified_pos_tags"] = SequenceLabelField(
                modified_pos_tags, text_field, "modified_pos_tags"
            )
        if "predicate_arg" in self.feature_labels:
            if predicate_arg_categories is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use predicate arg tags as "
                    " features. Pass them to text_to_instance."
                )
            fields["predicate_arg_tags"] = SequenceLabelField(
                predicate_arg_categories, text_field, "predicate_arg_tags"
            )

        # Add "tag label" to instance
        if self.tag_label == "ccg" and ccg_categories is not None:
            fields["tags"] = SequenceLabelField(ccg_categories, text_field, self.label_namespace)
        elif self.tag_label == "original_pos" and original_pos_tags is not None:
            fields["tags"] = SequenceLabelField(original_pos_tags, text_field, self.label_namespace)
        elif self.tag_label == "modified_pos" and modified_pos_tags is not None:
            fields["tags"] = SequenceLabelField(modified_pos_tags, text_field, self.label_namespace)
        elif self.tag_label == "predicate_arg" and predicate_arg_categories is not None:
            fields["tags"] = SequenceLabelField(
                predicate_arg_categories, text_field, self.label_namespace
            )

        return Instance(fields)
