# pylint: disable=no-self-use,invalid-name
import json
from unittest import TestCase

from allennlp.common import Params
from allennlp.common.params import replace_none
from allennlp.service.predictors.bidaf import BidafPredictor


class TestBidafPredictor(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        with open('tests/fixtures/bidaf/experiment.json') as f:
            config = json.loads(f.read())
            bidaf_config = Params(replace_none(config))

        model = BidafPredictor.from_config(bidaf_config)

        result = model.predict_json(inputs)

        assert "span_start_probs" in result
        assert "span_end_probs" in result
