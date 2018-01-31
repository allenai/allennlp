
from collections import OrderedDict
from typing import Dict, List
import logging
import os

from overrides import overrides
# NLTK is so performance orientated (ha ha) that they have lazy imports. Why? Who knows.
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader # pylint: disable=no-name-in-module
from nltk.tree import Tree

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SpanField, SequenceLabelField, ListField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ptb_trees")
class PennTreeBankDatasetReader(DatasetReader):
    """
    Reads constituency parses from the WSJ part of the Penn Tree Bank from the LDC.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        directory, filename = os.path.split(file_path)

        logger.info("Reading instances from lines in file at: %s", file_path)
        parses: List[Tree] = BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents()

        for parse in Tqdm.tqdm(parses):
            yield self.text_to_instance(parse.leaves(), [x[1] for x in parse.pos()], parse)

    @overrides
    def text_to_instance(self, # pylint: disable=arguments-differ
                         tokens: List[str],
                         pos_tags: List[str],
                         gold_tree: Tree = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        pos_tags ``List[str]``, required.
            The pos tags for the words in the sentence.
        gold_tree : ``Tree``, optional (default = None).
            The gold parse tree to create span labels from.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence.
            pos_tags : ``SequenceLabelField``
                The pos tags of the words in the sentence.
            spans : ``ListField[SpanField]``
                A ListField containing all possible subspans of the
                sentence.
            span_labels : ``SequenceLabelField``, optional.
                The constiutency tags for each of the possible spans, with
                respect to a gold parse tree. If a span is not contained
                within the tree, a span will have a ``NO-LABEL`` label.
        """
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        pos_tag_field = SequenceLabelField(pos_tags, text_field, "pos_tags")
        fields = {"tokens": text_field, "pos_tags": pos_tag_field}
        spans: List[Field] = []
        gold_labels = []

        if gold_tree is not None:
            gold_spans_with_pos_tags = self._get_gold_spans(gold_tree, 0, OrderedDict()).items()
            gold_spans = {span: label for (span, label)
                          in gold_spans_with_pos_tags if "-POS" not in label}
        else:
            gold_spans = None
        for start, end in enumerate_spans(tokens):
            spans.append(SpanField(start, end, text_field))

            if gold_spans is not None:
                if (start, end) in gold_spans.keys():
                    gold_labels.append(gold_spans[(start, end)])
                else:
                    gold_labels.append("NO-LABEL")

        span_list_field: ListField = ListField(spans)
        fields["spans"] = span_list_field
        if gold_tree is not None:
            fields["span_labels"] = SequenceLabelField(gold_labels, span_list_field)

        return Instance(fields)

    def _get_gold_spans(self, # pylint: disable=arguments-differ
                        tree: Tree,
                        index: int,
                        typed_spans: OrderedDict) -> OrderedDict:
        """
        Recursively construct the gold spans from an nltk ``Tree``.
        Spans are inclusive.

        Parameters
        ----------
        tree : ``Tree``, required.
            An NLTK parse tree to extract spans from.
        index : ``int``, required.
            The index of the current span in the sentence being considered.
        typed_spans : ``OrderedDict[Tuple[int, int], str]``, required.
            A ordered dictionary mapping spans to span labels.

        Returns
        -------
        typed_spans : ``OrderedDict[Tuple[int, int], str]``, required.
            A ordered dictionary mapping all subtree spans in the parse tree
            to their constituency labels. Leaf nodes have POS tag spans, which
            are denoted by a label of "LABEL-POS".
        """
        # NLTK leaves are strings.
        if isinstance(tree[0], str):
            # The "length" of a tree is defined by
            # NLTK as the number of children.
            # We don't actually want the spans for leaves, because
            # their labels are pos tags. However, it makes the
            # indexing more straightforward, so we'll collect them
            # and filter them out below. We subtract 1 from the end
            # index so the spans are inclusive.
            typed_spans[(index, index + len(tree) - 1)] = tree.label() + "-POS"
        else:
            # otherwise, the tree has children.
            first = True
            for child in tree:
                typed_spans = self._get_gold_spans(child, index, typed_spans)
                # OrderedDicts are implemented via a
                # doubly linked list, so this reversal is O(1).
                last_appended_span = next(reversed(typed_spans))
                # Set the end index of the current span to
                # the last appended index + 1, as the span was inclusive.
                index = last_appended_span[1] + 1
                if first:
                    start = last_appended_span[0]
                    first = False
            end = index
            typed_spans[(start, end - 1)] = tree.label()
        return typed_spans

    @classmethod
    def from_params(cls, params: Params) -> 'PennTreeBankDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return PennTreeBankDatasetReader(token_indexers=token_indexers)
