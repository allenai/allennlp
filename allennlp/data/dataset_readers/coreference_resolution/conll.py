import logging
import re
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, IndexField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class _DocumentState:
    """
    Represents the state of a document. Words are collected in a sentence buffer,
    which are incrementally collected in a list when sentences end, representing
    all tokens in the document.

    Additionally, this class contains a per-id stacks to hold the start indices of
    active spans (spans which we are inside of when processing a given word). Spans
    with the same id can be nested, which is why we collect these opening spans
    on a stack, e.g:

    [Greg, the baker who referred to [himself]_ID1 as 'the bread man']_ID1

    Once an active span is closed, the span is added to the cluster for the given id.
    """
    def __init__(self) -> None:
        self.sentence_buffer: List[str] = []
        self.sentences: List[List[str]] = []
        self.num_total_words = 0
        # Cluster id -> List of (start_index, end_index) spans.
        self.clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        # Cluster id -> List of start_indices which are open for this id.
        self.coref_stacks: DefaultDict[int, List[int]] = collections.defaultdict(list)

    def assert_document_is_finished(self) -> None:
        if self.sentence_buffer:
            raise ConfigurationError("Processing was attempted on a document which had "
                                     "incomplete sentences. Current sentence buffer "
                                     "contains: {}".format(self.sentence_buffer))

        if not self.sentences or self.num_total_words == 0:
            raise ConfigurationError("Processing was attempted on a document with no sentences or words.")

        if any(x for x in self.coref_stacks.values()):
            raise ConfigurationError("Processing was attempted on a document which contains incomplete "
                                     "(unclosed) spans within the scope of the document. Current ids "
                                     "and start indices" " of unclosed spans: {}".format(self.coref_stacks))

    def complete_sentence(self) -> None:
        self.sentences.append(self.sentence_buffer)
        self.sentence_buffer = []

    def add_word(self, word) -> None:
        self.sentence_buffer.append(word)
        self.num_total_words += 1

    def canonicalize_clusters(self) -> List[List[Tuple[int, int]]]:
        """
        The CONLL 2012 data includes 2 annotatated spans which are identical,
        but different ids. This checks all clusters for spans which are
        identical, and if it finds any, merges the clusters containing the
        identical spans.
        """
        merged_clusters: List[Set[Tuple[int, int]]] = []
        for cluster in self.clusters.values():
            cluster_with_overlapping_mention = None
            for mention in cluster:
                # Look at clusters we have already processed to
                # see if they contain a mention in the current
                # cluster for comparison.
                for cluster2 in merged_clusters:
                    if mention in cluster2:
                        # first cluster in merged clusters
                        # which contains this mention.
                        cluster_with_overlapping_mention = cluster2
                        break
                # Already encountered overlap - no need to keep looking.
                if cluster_with_overlapping_mention is not None:
                    break
            if cluster_with_overlapping_mention is not None:
                # Merge cluster we are currently processing into
                # the cluster in the processed list.
                cluster_with_overlapping_mention.update(cluster)
            else:
                merged_clusters.append(set(cluster))
        return [list(c) for c in merged_clusters]


@DatasetReader.register("coref")
class ConllCorefReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``span_starts``, a ``ListField[IndexField]`` of inclusive
    start indices for span candidates, ``span_ends``, a ``ListField[IndexField]`` of inclusive end
    indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._begin_document_regex = re.compile(r"#begin document \((.*)\); part (\d+)")

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        instances = []
        with open(file_path) as dataset_file:
            document_state = _DocumentState()

            for line in dataset_file:

                if self._begin_document_regex.match(line):
                    # We're beginning a document. Refresh the state.
                    document_state = _DocumentState()

                elif line.startswith("#end document"):
                    # We've finished a document.
                    document_state.assert_document_is_finished()
                    clusters = document_state.canonicalize_clusters()
                    instance = self.text_to_instance(document_state.sentences, clusters)
                    instances.append(instance)
                else:
                    # Process a line.
                    self._handle_line(line, document_state)

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            span_starts : ``ListField[IndexField]``
                A ListField containing the span starts represented as ``IndexFields``
                with respect to the document text.
            span_ends : ``ListField[IndexField]``
                A ListField containing the span ends represented as ``IndexFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``span_starts`` ``ListField``.

        """
        flattened_sentences = [token for sentence in sentences for token in sentence]

        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        span_starts: List[Field] = []
        span_ends: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        sentence_offset = 0
        for sentence in sentences:
            for start_index in range(len(sentence)):
                for end_index in range(start_index, min(start_index + self._max_span_width, len(sentence))):
                    start = sentence_offset + start_index
                    end = sentence_offset + end_index

                    if span_labels is not None:
                        if (start, end) in cluster_dict:
                            span_labels.append(cluster_dict[(start, end)])
                        else:
                            span_labels.append(-1)

                    span_starts.append(IndexField(start, text_field))
                    span_ends.append(IndexField(end, text_field))
            sentence_offset += len(sentence)

        span_starts_field = ListField(span_starts)
        span_ends_field = ListField(span_ends)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "span_starts": span_starts_field,
                                    "span_ends": span_ends_field,
                                    "metadata": metadata_field}
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_starts_field)

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> "ConllCorefReader":
        token_indexers = TokenIndexer.dict_from_params(params.pop("token_indexers", {}))
        max_span_width = params.pop("max_span_width")
        params.assert_empty(cls.__name__)
        return cls(token_indexers=token_indexers, max_span_width=max_span_width)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    def _handle_line(self, line: str, document_state: _DocumentState) -> None:
        row = line.split()
        if not row:
            # End of a sentence. Clear the sentence buffer and
            # add sentence to document sentences.
            document_state.complete_sentence()
        else:
            if len(row) < 12:
                raise ConfigurationError("Encountered a non-empty line with fewer than 12 entries"
                                         " - this does not match the CONLL format.: {}".format(row))
            word = self._normalize_word(row[3])
            coref = row[-1]
            word_index = document_state.num_total_words
            document_state.add_word(word)

            if coref != "-":
                for segment in coref.split("|"):
                    # The conll representation of coref spans allows spans to
                    # overlap. If spans end or begin at the same word, they are
                    # separated by a "|".
                    if segment[0] == "(":
                        # The span begins at this word.
                        if segment[-1] == ")":
                            # The span begins and ends at this word (single word span).
                            cluster_id = int(segment[1:-1])
                            document_state.clusters[cluster_id].append((word_index, word_index))
                        else:
                            # The span is starting, so we record the index of the word.
                            cluster_id = int(segment[1:])
                            document_state.coref_stacks[cluster_id].append(word_index)
                    else:
                        # The span for this id is ending, but didn't start at this word.
                        # Retrieve the start index from the document state and
                        # add the span to the clusters for this id.
                        cluster_id = int(segment[:-1])
                        start = document_state.coref_stacks[cluster_id].pop()
                        document_state.clusters[cluster_id].append((start, word_index))
