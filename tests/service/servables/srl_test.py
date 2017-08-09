# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.common import Params
from allennlp.service.servable.models.semantic_role_labeler import SemanticRoleLabelerServable


class TestSrlServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "The squirrel wrote a unit test to make sure its nuts worked as designed."
        }

        params = Params({
                "tokenizer": {
                        "type": "word"
                },
                "token_indexers": {
                        "tokens": {
                                "type": "single_id",
                                "lowercase_tokens" : True
                        }
                },
                "vocab_dir": "tests/fixtures/vocab_srl",
                "model": {
                        "type": "semantic_role_labeler",
                        "text_field_embedder": {
                                "tokens": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                }
                        },
                        "stacked_encoder": {
                                "type": "lstm",
                                "input_size": 6,
                                "hidden_size": 7,
                                "num_layers": 2
                        }
                }
        })

        model = SemanticRoleLabelerServable.from_params(params)

        result = model.predict_json(inputs)

        # TODO(joelgrus): update this when you figure out the result format
        assert result
