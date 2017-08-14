# pylint: disable=no-self-use,invalid-name
import json
from unittest import TestCase

from allennlp.common import Params
from allennlp.common.params import replace_none
from allennlp.service.servable.models.bidaf import BidafServable


class TestBidafServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "question": "What kind of test succeeded on its first attempt?",
                "passage": "One time I was writing a unit test, and it succeeded on the first attempt."
        }

        with open('experiment_config/bidaf.json') as f:
            config = json.loads(f.read())
            config['serialization_prefix'] = 'tests/fixtures/bidaf/serialization'
            config['model']['text_field_embedder']['tokens']['pretrained_file'] = \
                'tests/fixtures/glove.6B.100d.sample.txt.gz'
            bidaf_config = Params(replace_none(config))

        model = BidafServable.from_config(bidaf_config)

        result = model.predict_json(inputs)

        assert "span_start_probs" in result
        assert "span_end_probs" in result
