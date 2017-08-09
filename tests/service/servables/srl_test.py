# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from allennlp.service.servable.models.semantic_role_labeler import SemanticRoleLabelerServable


class TestSrlServable(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "The squirrel wrote a unit test to make sure its nuts worked as designed."
        }

        model = SemanticRoleLabelerServable()

        result = model.predict_json(inputs)

        # TODO(joelgrus): update this when you figure out the result format
        verbs = result.get("verbs")

        assert verbs  # is not None

        assert any(v["verb"] == "wrote" for v in verbs)
        assert any(v["verb"] == "make" for v in verbs)
        assert any(v["verb"] == "worked" for v in verbs)
