# pylint: disable=no-self-use,invalid-name
import json
from unittest import TestCase

from allennlp.common import Params, constants
from allennlp.common.params import replace_none
from allennlp.service.servable.models.decomposable_attention import DecomposableAttentionServable


class TestDecomposableAttentionServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        with open('experiment_config/decomposable_attention.json') as f:
            config = json.loads(f.read())
            config['trainer']['serialization_prefix'] = 'tests/fixtures/decomposable_attention'
            config['model']['text_field_embedder']['tokens']['pretrained_file'] = \
                    'tests/fixtures/glove.6B.300d.sample.txt.gz'
            decomposable_attention_config = Params(replace_none(config))


        model = DecomposableAttentionServable.from_config(decomposable_attention_config)

        result = model.predict_json(inputs)

        assert "label_probs" in result
