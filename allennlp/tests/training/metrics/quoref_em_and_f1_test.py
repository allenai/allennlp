# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import QuorefEmAndF1


class QuorefEmAndF1Test(AllenNlpTestCase):
    def test_metric(self):
        metric = QuorefEmAndF1()
        metric(["Test string", "another string"], ["Test", "different string"])
        assert metric.get_metric() == (0.0, 0.58)

    def test_metric_treats_strings_and_singleton_lists_the_same(self):
        metric1 = QuorefEmAndF1()
        metric2 = QuorefEmAndF1()
        metric1("string", ["Test", "different string"])
        metric2(["string"], ["Test", "different string"])
        assert metric1.get_metric() == metric2.get_metric()
