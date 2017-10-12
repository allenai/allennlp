# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers import ConllCorefReader
from allennlp.common.testing import AllenNlpTestCase


class TestCorefReader(AllenNlpTestCase):
    def test_read_from_file(self):
        conll_reader = ConllCorefReader(max_span_width=10)
        dataset = conll_reader.read('tests/fixtures/data/coref/sample.gold_conll')

