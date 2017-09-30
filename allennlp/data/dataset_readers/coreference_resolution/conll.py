import logging
import re
import collections
from typing import Dict, List, Tuple

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

@DatasetReader.register("coref")
class ConllCorefReader(DatasetReader):
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._begin_doc_regex = re.compile(r"#begin document \((.*)\); part (\d+)")

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        instances = []
        with open(file_path) as dataset_file:
            document_state = DocumentState()
            for line in dataset_file.readlines():
                if self.handle_line(line, document_state):
                    clusters = document_state.finalize()
                    instance = self.text_to_instance(document_state.doc_key,
                                                     document_state.sentences,
                                                     clusters)
                    instances.append(instance)
                    document_state = DocumentState()
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         doc_key: str,
                         sentences: List[List[str]],
                         clusters: List[List[Tuple[int, int]]]) -> Instance:
        # pylint: disable=arguments-differ
        flattened_sentences = [t for s in sentences for t in s]
        text_field = TextField([Token(t) for t in flattened_sentences], self._token_indexers)

        cluster_dict = {}
        for cluster_id, cluster in enumerate(clusters):
            assert len(cluster) > 1
            for mention in cluster:
                cluster_dict[tuple(mention)] = cluster_id

        span_labels = []
        span_starts = []
        span_ends = []
        sentence_offset = 0
        for sentence in sentences:
            for i in range(len(sentence)):
                for j in range(i, min(i + self._max_span_width, len(sentence))):
                    start = sentence_offset + i
                    end = sentence_offset + j
                    if (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)
                    start_field: Field = IndexField(start, text_field)
                    end_field: Field = IndexField(end, text_field)
                    span_starts.append(start_field)
                    span_ends.append(end_field)
            sentence_offset += len(sentence)

        span_starts_field = ListField(span_starts)
        span_ends_field = ListField(span_ends)
        span_labels_field = SequenceLabelField(span_labels, span_starts_field)

        metadata_field = MetadataField({"doc_key" : doc_key,
                                        "clusters": clusters,
                                        "text": flattened_sentences})

        fields: Dict[str, Field] = {'text': text_field,
                                    'span_starts': span_starts_field,
                                    'span_ends': span_ends_field,
                                    'span_labels': span_labels_field,
                                    'metadata': metadata_field}
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'ConllCorefReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        max_span_width = params.pop('max_span_width')
        params.assert_empty(cls.__name__)
        return cls(token_indexers=token_indexers, max_span_width=max_span_width)

    @staticmethod
    def get_doc_key(doc_id, part):
        return "{}_{}".format(doc_id, int(part))

    @staticmethod
    def normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

    def handle_line(self, line, document_state):
        begin_document_match = self._begin_doc_regex.match(line)
        if begin_document_match:
            document_state.assert_empty()
            document_state.doc_key = self.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
            return False
        elif line.startswith("#end document"):
            document_state.assert_finalizable()
            return True
        else:
            row = line.split()
            if not row:
                document_state.sentences.append(tuple(document_state.text))
                del document_state.text[:]
                return False
            assert len(row) >= 12

            word = self.normalize_word(row[3])
            coref = row[-1]

            word_index = len(document_state.text) + sum(len(s) for s in document_state.sentences)
            document_state.text.append(word)

            if coref != "-":
                for segment in coref.split("|"):
                    if segment[0] == "(":
                        if segment[-1] == ")":
                            cluster_id = int(segment[1:-1])
                            document_state.clusters[cluster_id].append((word_index, word_index))
                        else:
                            cluster_id = int(segment[1:])
                            document_state.coref_stacks[cluster_id].append(word_index)
                    else:
                        cluster_id = int(segment[:-1])
                        start = document_state.coref_stacks[cluster_id].pop()
                        document_state.clusters[cluster_id].append((start, word_index))
            return False

class DocumentState(object):
    def __init__(self):
        self.doc_key = None
        self.text = []
        self.sentences = []
        self.clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)

    def assert_empty(self):
        assert self.doc_key is None
        assert not self.text
        assert not self.sentences
        assert not self.coref_stacks
        assert not self.clusters

    def assert_finalizable(self):
        assert self.doc_key is not None
        assert not self.text
        assert self.sentences
        assert all(not s for s in self.coref_stacks.values())

    def finalize(self):
        merged_clusters = []
        for cluster1 in self.clusters.values():
            existing = None
            for mention in cluster1:
                for cluster2 in merged_clusters:
                    if mention in cluster2:
                        existing = cluster2
                        break
                if existing is not None:
                    break
            if existing is not None:
                existing.update(cluster1)
            else:
                merged_clusters.append(set(cluster1))
        return [tuple(c) for c in merged_clusters]
