from typing import Any, Dict, List, Optional, Tuple, Set

from allennlp.data.fields import (
    Field,
    ListField,
    TextField,
    SpanField,
    MetadataField,
    SequenceLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans


def make_coref_instance(
    sentences: List[List[str]],
    token_indexers: Dict[str, TokenIndexer],
    max_span_width: int,
    gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
    wordpiece_modeling_tokenizer: PretrainedTransformerTokenizer = None,
) -> Instance:

    """
    # Parameters

    sentences : `List[List[str]]`, required.
        A list of lists representing the tokenised words and sentences in the document.
    gold_clusters : `Optional[List[List[Tuple[int, int]]]]`, optional (default = None)
        A list of all clusters in the document, represented as word spans with absolute indices
        in the entire document. Each cluster contains some number of spans, which can be nested
        and overlap. If there are exact matches between clusters, they will be resolved
        using `_canonicalize_clusters`.

    # Returns

    An `Instance` containing the following `Fields`:
        text : `TextField`
            The text of the full document.
        spans : `ListField[SpanField]`
            A ListField containing the spans represented as `SpanFields`
            with respect to the document text.
        span_labels : `SequenceLabelField`, optional
            The id of the cluster which each possible span belongs to, or -1 if it does
                not belong to a cluster. As these labels have variable length (it depends on
                how many spans we are considering), we represent this a as a `SequenceLabelField`
                with respect to the `spans `ListField`.
    """
    flattened_sentences = [_normalize_word(word) for sentence in sentences for word in sentence]

    if wordpiece_modeling_tokenizer is not None:
        flat_sentences_tokens, offsets = wordpiece_modeling_tokenizer.intra_word_tokenize(
            flattened_sentences
        )
        flattened_sentences = [t.text for t in flat_sentences_tokens]
    else:
        flat_sentences_tokens = [Token(word) for word in flattened_sentences]

    text_field = TextField(flat_sentences_tokens, token_indexers)

    cluster_dict = {}
    if gold_clusters is not None:
        gold_clusters = _canonicalize_clusters(gold_clusters)

        if wordpiece_modeling_tokenizer is not None:
            for cluster in gold_clusters:
                for mention_id, mention in enumerate(cluster):
                    start = offsets[mention[0]][0]
                    end = offsets[mention[1]][1]
                    cluster[mention_id] = (start, end)

        for cluster_id, cluster in enumerate(gold_clusters):
            for mention in cluster:
                cluster_dict[tuple(mention)] = cluster_id

    spans: List[Field] = []
    span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

    sentence_offset = 0
    for sentence in sentences:
        for start, end in enumerate_spans(
            sentence, offset=sentence_offset, max_span_width=max_span_width
        ):
            if wordpiece_modeling_tokenizer is not None:
                start = offsets[start][0]
                end = offsets[end][1]

                # `enumerate_spans` uses word-level width limit; here we apply it to wordpieces
                # We have to do this check here because we use a span width embedding that has
                # only `max_span_width` entries, and since we are doing wordpiece
                # modeling, the span width embedding operates on wordpiece lengths. So a check
                # here is necessary or else we wouldn't know how many entries there would be.
                if end - start + 1 > max_span_width:
                    continue
                # We also don't generate spans that contain special tokens
                if start < wordpiece_modeling_tokenizer.num_added_start_tokens:
                    continue
                if (
                    end
                    >= len(flat_sentences_tokens)
                    - wordpiece_modeling_tokenizer.num_added_end_tokens
                ):
                    continue

            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)

            spans.append(SpanField(start, end, text_field))
        sentence_offset += len(sentence)

    span_field = ListField(spans)

    metadata: Dict[str, Any] = {"original_text": flattened_sentences}
    if gold_clusters is not None:
        metadata["clusters"] = gold_clusters
    metadata_field = MetadataField(metadata)

    fields: Dict[str, Field] = {
        "text": text_field,
        "spans": span_field,
        "metadata": metadata_field,
    }
    if span_labels is not None:
        fields["span_labels"] = SequenceLabelField(span_labels, span_field)

    return Instance(fields)


def _normalize_word(word):
    if word in ("/.", "/?"):
        return word[1:]
    else:
        return word


def _canonicalize_clusters(clusters: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The data might include 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters:
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
