# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models import Model
from allennlp.service.predictors import PredictorCollection

class TestPredictors(AllenNlpTestCase):
    def test_predictor_functionality(self):
        predictors = PredictorCollection()

        assert not predictors.list_available()  # should be empty

        class FakeModel(Model):  # pylint: disable=abstract-method
            def __init__(self):  # pylint: disable=super-init-not-called
                pass

        fake1 = FakeModel()
        fake2 = FakeModel()

        predictors.register('fake1', fake1)
        predictors.register('fake2', fake2)

        assert predictors.get('fake1') is fake1
        assert predictors.get('fake2') is fake2
        assert predictors.get('fake3') is None

        assert set(predictors.list_available()) == {'fake1', 'fake2'}
