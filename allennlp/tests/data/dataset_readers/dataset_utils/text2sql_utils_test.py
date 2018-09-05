# pylint: disable=no-self-use,invalid-name,protected-access

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils import text2sql_utils

class Text2SqlUtilsTest(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        self.data = AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants_tiny.json'

    def test_data(self):
        print(text2sql_utils.get_split(str(self.data), "train", cross_validation_split=3))