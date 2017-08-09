# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.service.servable.models.bidaf import BidafServable


class TestBidafServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        model = BidafServable()

        result = model.predict_json(inputs)

        assert "span_start_probs" in result
        assert "span_end_probs" in result
