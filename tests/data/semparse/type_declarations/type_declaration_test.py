# pylint: disable=no-self-use
from allennlp.data.semparse.type_declarations import type_declaration as types
from allennlp.common.testing import AllenNlpTestCase


class TestTypeResolution(AllenNlpTestCase):
    def test_basic_types_conflict_on_names(self):
        type_a = types.NamedBasicType("A")
        type_b = types.NamedBasicType("B")
        assert type_a.resolve(type_b) is None
