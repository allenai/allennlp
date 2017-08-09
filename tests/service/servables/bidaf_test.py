# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.common.params import Params
from allennlp.service.servable.models.bidaf import BidafServable


class TestBidafServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        bidaf_params = Params({
                "glove_path": "tests/fixtures/glove.6B.100d.sample.txt.gz",
                "tokenizer": {
                        "type": "word"
                },
                "token_indexers": {
                        "tokens": {
                                "type": "single_id",
                                "lowercase_tokens" : True
                        },
                        "token_characters": {
                                "type": "characters"
                        }
                },
                "vocab_dir": "tests/fixtures/vocab_bidaf",
                "model": {
                        "type": "bidaf"
                }
        })

        model = BidafServable.from_params(bidaf_params)

        result = model.predict_json(inputs)

        assert "span_start_probs" in result
        assert "span_end_probs" in result
