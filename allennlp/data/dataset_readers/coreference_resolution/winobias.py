import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict

from overrides import overrides

from allennlp.common import Params
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
    on another corpora.

    Returns a list of ``Instances`` which have four fields: ``text``, a ``TextField``
    containing the full sentence text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
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
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        for sentence in open(file_path, "r"):
            tokens = sentence.strip().split(" ")
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
            words = []
            inside_square_span = False
            inside_round_span = False
            for index, token in enumerate(tokens):
                # Coreference is annotated using [square brackets]
                # or (round brackets) around coreferent phrases.
                if "[" in token and "]" in token:
                    clusters[0].append((index, index))
                elif "[" in token:
                    clusters[0].append((index, index))
                    inside_square_span = True
                elif inside_square_span or "]" in token:
                    old_span = clusters[0][-1]
                    clusters[0][-1] = (old_span[0], old_span[1] + 1)
                    inside_square_span = False

                if "(" in token and ")" in token:
                    clusters[1].append((index, index))
                elif "(" in token:
                    clusters[1].append((index, index))
                    inside_round_span = True
                elif inside_round_span or ")" in token:
                    old_span = clusters[1][-1]
                    clusters[1][-1] = (old_span[0], old_span[1] + 1)
                    inside_round_span = False

                if token.endswith("."):
                    # Winobias is tokenised, but not for full stops.
                    # We'll just special case them here.
                    token = token[:-1]
                    words.append(token.strip("[]()"))
                    words.append(".")
                else:
                    words.append(token.strip("[]()"))

            yield self.text_to_instance(words, [x for x in clusters.values()])

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: List[str],
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
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        metadata: Dict[str, Any] = {"original_text": sentence}
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField([Token(word) for word in sentence], self._token_indexers)

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

    @classmethod
    def from_params(cls, params: Params) -> "WinobiasReader":
        token_indexers = TokenIndexer.dict_from_params(params.pop("token_indexers", {}))
        max_span_width = params.pop_int("max_span_width")
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(token_indexers=token_indexers, max_span_width=max_span_width, lazy=lazy)
