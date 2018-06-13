# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase

class TestUniversalDependenciesDatasetReader(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "dependencies.conllu"

    def test_read_from_file(self):

        reader = UniversalDependenciesDatasetReader()

        instances = list(reader.read(str(self.data_path)))

        for instance in instances:
            print(instance)