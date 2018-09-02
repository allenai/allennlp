# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers.semantic_dependency_parsing import SemanticDependenciesDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestSemanticDependencyParsingDatasetReader:
    def test_read_from_file(self):
        reader = SemanticDependenciesDatasetReader()
        instances = reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'dm.sdp')
        instances = ensure_list(instances)

        for instance in instances:
            print(instance)
