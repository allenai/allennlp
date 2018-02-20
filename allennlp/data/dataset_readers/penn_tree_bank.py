
from typing import Dict, List, Tuple
import logging
import os

from overrides import overrides
# NLTK is so performance orientated (ha ha) that they have lazy imports. Why? Who knows.
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader # pylint: disable=no-name-in-module
from nltk.tree import Tree

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SpanField, SequenceLabelField, ListField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ptb_trees")
class PennTreeBankConstituencySpanDatasetReader(DatasetReader):
    """
    Reads constituency parses from the WSJ part of the Penn Tree Bank from the LDC.
    This ``DatasetReader`` is designed for use with a span labelling model, so
    it enumerates all possible spans in the sentence and returns them, along with gold
    labels for the relevant spans present in a gold tree, if provided.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    use_pos_tags : ``bool``, optional, (default = ``True``)
        Whether or not the instance should contain gold POS tags
        as a field.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_pos_tags: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_pos_tags = use_pos_tags

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        directory, filename = os.path.split(file_path)
        logger.info("Reading instances from lines in file at: %s", file_path)

        for parse in BracketParseCorpusReader(root=directory, fileids=[filename]).parsed_sents():
            pos_tags = [x[1] for x in parse.pos()] if self._use_pos_tags else None
            yield self.text_to_instance(parse.leaves(), pos_tags, parse)

    @overrides
    def text_to_instance(self, # type: ignore
                         tokens: List[str],
                         pos_tags: List[str] = None,
                         gold_tree: Tree = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        pos_tags ``List[str]``, optional, (default = None).
            The POS tags for the words in the sentence.
        gold_tree : ``Tree``, optional (default = None).
            The gold parse tree to create span labels from.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence.
            pos_tags : ``SequenceLabelField``
                The POS tags of the words in the sentence.
                Only returned if ``use_pos_tags`` is ``True``
            spans : ``ListField[SpanField]``
                A ListField containing all possible subspans of the
                sentence.
            span_labels : ``SequenceLabelField``, optional.
                The constiutency tags for each of the possible spans, with
                respect to a gold parse tree. If a span is not contained
                within the tree, a span will have a ``NO-LABEL`` label.
            gold_tree : ``MetadataField(Tree)``
                The gold NLTK parse tree for use in evaluation.
        """
        # pylint: disable=arguments-differ
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        if self._use_pos_tags and pos_tags is not None:
            pos_tag_field = SequenceLabelField(pos_tags, text_field, "pos_tags")
            fields["pos_tags"] = pos_tag_field
        elif self._use_pos_tags:
            raise ConfigurationError("use_pos_tags was set to True but no gold pos"
                                     " tags were passed to the dataset reader.")
        spans: List[Field] = []
        gold_labels = []

        if gold_tree is not None:
            gold_spans_with_pos_tags: Dict[Tuple[int, int], str] = {}
            self._get_gold_spans(gold_tree, 0, gold_spans_with_pos_tags)
            gold_spans = {span: label for (span, label)
                          in gold_spans_with_pos_tags.items() if "-POS" not in label}
        else:
            gold_spans = None
        for start, end in enumerate_spans(tokens):
            spans.append(SpanField(start, end, text_field))

            if gold_spans is not None:
                if (start, end) in gold_spans.keys():
                    gold_labels.append(gold_spans[(start, end)])
                else:
                    gold_labels.append("NO-LABEL")

        if gold_tree:
            fields["gold_tree"] = MetadataField(gold_tree)

        span_list_field: ListField = ListField(spans)
        fields["spans"] = span_list_field
        if gold_tree is not None:
            fields["span_labels"] = SequenceLabelField(gold_labels, span_list_field)

        return Instance(fields)

    def _get_gold_spans(self, # pylint: disable=arguments-differ
                        tree: Tree,
                        index: int,
                        typed_spans: Dict[Tuple[int, int], str]) -> int:
        """
        Recursively construct the gold spans from an nltk ``Tree``.
        Spans are inclusive.

        Parameters
        ----------
        tree : ``Tree``, required.
            An NLTK parse tree to extract spans from.
        index : ``int``, required.
            The index of the current span in the sentence being considered.
        typed_spans : ``Dict[Tuple[int, int], str]``, required.
            A dictionary mapping spans to span labels.

        Returns
        -------
        typed_spans : ``Dict[Tuple[int, int], str]``.
            A dictionary mapping all subtree spans in the parse tree
            to their constituency labels. Leaf nodes have POS tag spans, which
            are denoted by a label of "LABEL-POS".
        """
        # NLTK leaves are strings.
        if isinstance(tree[0], str):
            # The "length" of a tree is defined by
            # NLTK as the number of children.
            # We don't actually want the spans for leaves, because
            # their labels are POS tags. However, it makes the
            # indexing more straightforward, so we'll collect them
            # and filter them out below. We subtract 1 from the end
            # index so the spans are inclusive.
            end = index + len(tree)
            typed_spans[(index, end - 1)] = tree.label() + "-POS"
        else:
            # otherwise, the tree has children.
            child_start = index
            for child in tree:
                # typed_spans is being updated inplace.
                end = self._get_gold_spans(child, child_start, typed_spans)
                child_start = end
            # Set the end index of the current span to
            # the last appended index - 1, as the span is inclusive.
            typed_spans[(index, end - 1)] = tree.label()
        return end

    @classmethod
    def from_params(cls, params: Params) -> 'PennTreeBankConstituencySpanDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        use_pos_tags = params.pop('use_pos_tags', True)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return PennTreeBankConstituencySpanDatasetReader(token_indexers=token_indexers,
                                                         use_pos_tags=use_pos_tags,
                                                         lazy=lazy)
