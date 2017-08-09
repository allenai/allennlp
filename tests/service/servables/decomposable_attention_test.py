# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.service.servable.models.decomposable_attention import DecomposableAttentionServable


class TestDecomposableAttentionServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        model = DecomposableAttentionServable()

        result = model.predict_json(inputs)

        assert "label_probs" in result
