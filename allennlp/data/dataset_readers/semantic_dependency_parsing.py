from typing import Dict, List, Tuple
import logging
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import AdjacencyField, MetadataField, SequenceLabelField
from allennlp.data.fields import Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)

FIELDS = ["id", "form", "lemma", "pos", "head", "deprel", "top", "pred", "frame"]

def parse_sentence(sentence: str):
    annotated_sentence = []
    arc_indices = []
    arc_labels = []
    preds = []

    lines = [line.split("\t") for line in sentence.split("\n")
             if line and not line.strip().startswith("#")]
    for line_idx, line in enumerate(lines):
        annotated_token = { k:v for k, v in zip(FIELDS, line)}
        if annotated_token['pred'] == "+":
            preds.append(line_idx)
        annotated_sentence.append(annotated_token)

    for line_idx, line in enumerate(lines):
        for pred_idx, arg in enumerate(line[len(FIELDS):]):
            if arg != "_":
                arc_indices.append((line_idx, preds[pred_idx]))
                arc_labels.append(arg)
    return annotated_sentence, arc_indices, arc_labels


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield parse_sentence(sentence)


@DatasetReader.register("semantic_dependencies")
class SemanticDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the SemEval 2015 Task 18 (Broad-coverage Semantic Dependency Parsing)
    format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading semantic dependency parsing data from: %s", file_path)

        with open(file_path) as sdp_file:
            for annotated_sentence, directed_arc_indices, arc_labels in lazy_parse(sdp_file.read()):
                # If there are no arc indices, skip this instance.
                if not directed_arc_indices:
                    continue
                tokens = [word["form"] for word in annotated_sentence]
                pos_tags = [word["pos"] for word in annotated_sentence]

                # TODO: make adjacency list / matrix of arcs and labels.
                yield self.text_to_instance(tokens, pos_tags, directed_arc_indices, arc_labels)

    @overrides
    def text_to_instance(self,
                         tokens: List[str],
                         pos_tags: List[str] = None,
                         arc_indices: List[Tuple[int, int]] = None,
                         arc_labels: List[str] = None) -> Instance:

        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in tokens], self._token_indexers)
        fields["tokens"] = token_field
        fields["metadata"] = MetadataField({"tokens": tokens})
        if pos_tags is not None:
            fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos")
        if arc_indices is not None and arc_labels is not None:
            fields["arc_labels"] = AdjacencyField(arc_indices, token_field, arc_labels)

        return Instance(fields)
