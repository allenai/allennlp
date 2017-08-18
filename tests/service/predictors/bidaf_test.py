# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.common import Params
from allennlp.service.predictors.bidaf import BidafPredictor


class TestBidafPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        bidaf_config = Params.from_file('tests/fixtures/bidaf/experiment.json')
        model = BidafPredictor.from_config(bidaf_config)

        result = model.predict_json(inputs)

        assert "span_start_probs" in result
        assert "span_end_probs" in result
