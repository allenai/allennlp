# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.service.servable.models.bidaf import BidafServable

import pytest

class TestBidafServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "Is this a test?",
                "passage": "This is definitely a test"
        }

        model = BidafServable()

        result = model.predict_json(inputs)

        assert "span_start_probs" in result
        assert "span_end_probs" in result
