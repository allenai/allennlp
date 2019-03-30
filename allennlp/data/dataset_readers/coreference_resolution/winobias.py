import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import  enumerate_spans

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("winobias")
class WinobiasReader(DatasetReader):
    """
    TODO(Mark): Add paper reference.

    Winobias is a dataset to analyse the issue of gender bias in co-reference
    resolution. It contains simple sentences with pro/anti stereotypical gender
    associations with which to measure the bias of a coreference system trained
    on another corpus. It is effectively a toy dataset and as such, uses very
    simplistic language; it has little use outside of evaluating a model for bias.

    The dataset is formatted with a single sentence per line, with a maximum of 2
    non-nested coreference clusters annotated using either square or round brackets.
    For example:

    [The salesperson] sold (some books) to the librarian because [she] was trying to sell (them).


    Returns a list of ``Instances`` which have four fields: ``text``, a ``TextField``
    containing the full sentence text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate in the ``metadata`` also.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the sentence.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):

        for sentence in open(cached_path(file_path), "r"):
            tokens = sentence.strip().split(" ")
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
            words = []
            for index, token in enumerate(tokens):
                # Coreference is annotated using [square brackets]
                # or (round brackets) around coreferent phrases.
                if "[" in token and "]" in token:
                    clusters[0].append((index, index))
                elif "[" in token:
                    clusters[0].append((index, index))
                elif "]" in token:
                    old_span = clusters[0][-1]
                    clusters[0][-1] = (old_span[0], index)

                if "(" in token and ")" in token:
                    clusters[1].append((index, index))
                elif "(" in token:
                    clusters[1].append((index, index))
                elif ")" in token:
                    old_span = clusters[1][-1]
                    clusters[1][-1] = (old_span[0], index)

                if token.endswith("."):
                    # Winobias is tokenised, but not for full stops.
                    # We'll just special case them here.
                    token = token[:-1]
                    words.append(token.strip("[]()"))
                    words.append(".")
                else:
                    words.append(token.strip("[]()"))

            yield self.text_to_instance([Token(x) for x in words], [x for x in clusters.values()])

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: List[Token],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentence : ``List[Token]``, required.
            The already tokenised sentence to analyse.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the sentence, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full sentence.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the sentence text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        metadata: Dict[str, Any] = {"original_text": sentence}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField(sentence, self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        for start, end in enumerate_spans(sentence, max_span_width=self._max_span_width):
            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)

            spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "spans": span_field,
                                    "metadata": metadata_field}
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)
