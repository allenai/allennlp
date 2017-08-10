# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.common import Params, constants
from allennlp.service.servable.models.bidaf import BidafServable


class TestBidafServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        bidaf_config = Params({
                "dataset_reader": {
                        "token_indexers": {
                                "tokens": {
                                        "type": "single_id",
                                        "lowercase_tokens" : True
                                },
                                "token_characters": {
                                        "type": "characters"
                                }
                        }
                },
                "serialization_prefix": "tests/fixtures/bidaf",
                "model": {
                        "type": "bidaf"
                }
        })

        constants.GLOVE_PATH = 'tests/fixtures/glove.6B.100d.sample.txt.gz'

        model = BidafServable.from_config(bidaf_config)

        result = model.predict_json(inputs)

        assert "span_start_probs" in result
        assert "span_end_probs" in result
