# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestUniversalDependenciesDatasetReader(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "dependencies.conllu"

    def test_read_from_file(self):

        reader = UniversalDependenciesDatasetReader()

        instances = reader.read(str(self.data_path))