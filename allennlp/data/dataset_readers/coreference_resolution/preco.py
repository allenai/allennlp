import json
import logging
from typing import Dict, List, Optional, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.coreference_resolution.util import make_coref_instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("preco")
class PrecoReader(DatasetReader):
    """
    Reads a single JSON-lines file for [the PreCo dataset](https://www.aclweb.org/anthology/D18-1016.pdf).
    Each line contains a "sentences" key for a list of sentences and a "mention_clusters" key
    for the clusters.

    Returns a `Dataset` where the `Instances` have four fields : `text`, a `TextField`
    containing the full document text, `spans`, a `ListField[SpanField]` of inclusive start and
    end indices for span candidates, and `metadata`, a `MetadataField` that stores the instance's
    original text. For data with gold cluster labels, we also include the original `clusters`
    (a list of list of index pairs) and a `SequenceLabelField` of cluster ids for every span
    candidate.

    # Parameters

    max_span_width : `int`, required.
        The maximum width of candidate spans to consider.
    token_indexers : `Dict[str, TokenIndexer]`, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = None)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    """

    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        wordpiece_modeling_tokenizer: Optional[PretrainedTransformerTokenizer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._wordpiece_modeling_tokenizer = wordpiece_modeling_tokenizer

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as preco_file:
            for line in preco_file:
                example = json.loads(line)
                yield self.text_to_instance(example["sentences"], example["mention_clusters"])

    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentences: List[List[str]],
        gold_clusters: Optional[List[List[Tuple[int, int, int]]]] = None,
    ) -> Instance:
        sentence_offsets = [0]
        for sentence in sentences[:-1]:
            sent_length = len(sentence)
            if sentence == [" "]:  # paragraph separator
                sent_length = 0  # we ignore them
            sentence_offsets.append(sentence_offsets[-1] + sent_length)

        sentences = [sentence for sentence in sentences if sentence != [" "]]

        # Convert (sent_idx, rel_start, rel_exclusive_end) to (abs_start, abs_inclusive_end)
        for cluster in gold_clusters:
            for mention_id, (sent_idx, start, end) in enumerate(cluster):
                start = start + sentence_offsets[sent_idx]
                end = end + sentence_offsets[sent_idx] - 1  # exclusive -> inclusive
                cluster[mention_id] = (start, end)  # type: ignore

        return make_coref_instance(
            sentences,
            self._token_indexers,
            self._max_span_width,
            gold_clusters,  # type: ignore
            self._wordpiece_modeling_tokenizer,
        )
