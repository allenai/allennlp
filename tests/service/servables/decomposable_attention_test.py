# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.common import Params
from allennlp.service.servable.models.decomposable_attention import DecomposableAttentionServable


class TestDecomposableAttentionServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        params = Params({
                "glove_path": "tests/fixtures/glove.6B.300d.sample.txt.gz",
                "tokenizer": {
                        "type": "word"
                },
                "token_indexers": {
                        "tokens": {
                                "type": "single_id",
                                "lowercase_tokens" : True
                        }
                },
                "vocab_dir": "tests/fixtures/vocab_snli",
                "model": {
                        "type": "decomposable_attention"
                }
        })

        model = DecomposableAttentionServable.from_params(params)

        result = model.predict_json(inputs)

        assert "label_probs" in result
