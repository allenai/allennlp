from allennlp.data.dataset_readers.semantic_dependency_parsing import (
    SemanticDependenciesDatasetReader,
)
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestSemanticDependencyParsingDatasetReader:
    def test_read_from_file(self):
        reader = SemanticDependenciesDatasetReader()
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / "data" / "dm.sdp")
        instances = ensure_list(instances)

        instance = instances[0]
        arcs = instance.fields["arc_tags"]
        tokens = [x.text for x in instance.fields["tokens"].tokens]
        assert tokens == [
            "Pierre",
            "Vinken",
            ",",
            "61",
            "years",
            "old",
            ",",
            "will",
            "join",
            "the",
            "board",
            "as",
            "a",
            "nonexecutive",
            "director",
            "Nov.",
            "29",
            ".",
        ]
        assert arcs.indices == [
            (1, 0),
            (1, 5),
            (1, 8),
            (4, 3),
            (5, 4),
            (8, 11),
            (8, 16),
            (10, 8),
            (10, 9),
            (14, 11),
            (14, 12),
            (14, 13),
            (16, 15),
        ]
        assert arcs.labels == [
            "compound",
            "ARG1",
            "ARG1",
            "ARG1",
            "measure",
            "ARG1",
            "loc",
            "ARG2",
            "BV",
            "ARG2",
            "BV",
            "ARG1",
            "of",
        ]

        instance = instances[1]
        arcs = instance.fields["arc_tags"]
        tokens = [x.text for x in instance.fields["tokens"].tokens]
        assert tokens == [
            "Mr.",
            "Vinken",
            "is",
            "chairman",
            "of",
            "Elsevier",
            "N.V.",
            ",",
            "the",
            "Dutch",
            "publishing",
            "group",
            ".",
        ]
        assert arcs.indices == [
            (1, 0),
            (1, 2),
            (3, 2),
            (3, 4),
            (5, 4),
            (5, 6),
            (5, 11),
            (11, 8),
            (11, 9),
            (11, 10),
        ]
        assert arcs.labels == [
            "compound",
            "ARG1",
            "ARG2",
            "ARG1",
            "ARG2",
            "compound",
            "appos",
            "BV",
            "ARG1",
            "compound",
        ]
