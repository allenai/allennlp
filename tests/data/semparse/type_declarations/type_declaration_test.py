# pylint: disable=no-self-use,invalid-name
from nltk.sem.logic import ComplexType

from allennlp.data.semparse.type_declarations import type_declaration as types
from allennlp.data.semparse.type_declarations import wikitables_type_declaration as wt_types
from allennlp.common.testing import AllenNlpTestCase


class TestTypeResolution(AllenNlpTestCase):
    def test_basic_types_conflict_on_names(self):
        type_a = types.NamedBasicType("A")
        type_b = types.NamedBasicType("B")
        assert type_a.resolve(type_b) is None

    def test_get_valid_actions(self):
        type_r = types.NamedBasicType("R")
        type_d = types.NamedBasicType("D")
        type_e = types.NamedBasicType("E")
        name_mapping = {'sample_function': 'F'}
        # <e,<r,<d,r>>>
        type_signatures = {'F': ComplexType(type_e, ComplexType(type_r, ComplexType(type_d, type_r)))}
        basic_types = {type_r, type_d, type_e}
        valid_actions = types.get_valid_actions(name_mapping, type_signatures, basic_types)
        assert len(valid_actions) == 5
        assert valid_actions["<e,<r,<d,r>>>"] == {"<e,<r,<d,r>>> -> sample_function"}
        assert valid_actions["<r,<d,r>>"] == {"<r,<d,r>> -> [<e,<r,<d,r>>>, e]"}
        assert valid_actions["<d,r>"] == {"<d,r> -> [<r,<d,r>>, r]"}
        assert valid_actions["r"] == {"r -> [<d,r>, d]"}
        assert valid_actions["@START@"] == {"@START@ -> d", "@START@ -> r", "@START@ -> e"}

    def test_get_valid_actions_with_placeholder_type(self):
        type_r = types.NamedBasicType("R")
        type_d = types.NamedBasicType("D")
        type_e = types.NamedBasicType("E")
        name_mapping = {'sample_function': 'F'}
        # <#1,#1>
        type_signatures = {'F': types.IdentityType(types.ANY_TYPE, types.ANY_TYPE)}
        basic_types = {type_r, type_d, type_e}
        valid_actions = types.get_valid_actions(name_mapping, type_signatures, basic_types)
        assert len(valid_actions) == 5
        assert valid_actions["<#1,#1>"] == {"<#1,#1> -> sample_function"}
        assert valid_actions["e"] == {"e -> [<#1,#1>, e]"}
        assert valid_actions["r"] == {"r -> [<#1,#1>, r]"}
        assert valid_actions["d"] == {"d -> [<#1,#1>, d]"}
        assert valid_actions["@START@"] == {"@START@ -> d", "@START@ -> r", "@START@ -> e"}

    def test_get_valid_actions_with_any_type(self):
        type_r = types.NamedBasicType("R")
        type_d = types.NamedBasicType("D")
        type_e = types.NamedBasicType("E")
        name_mapping = {'sample_function': 'F'}
        # The purpose of this test is to ensure that ANY_TYPE gets substituted by every possible basic type,
        # to simulate an intermediate step while getting actions for a placeholder type.
        # I do not foresee defining a function type with ANY_TYPE. We should just use a ``PlaceholderType``
        # instead.
        # <?,r>
        type_signatures = {'F': ComplexType(types.ANY_TYPE, type_r)}
        basic_types = {type_r, type_d, type_e}
        valid_actions = types.get_valid_actions(name_mapping, type_signatures, basic_types)
        assert len(valid_actions) == 5
        assert valid_actions["<d,r>"] == {"<d,r> -> sample_function"}
        assert valid_actions["<e,r>"] == {"<e,r> -> sample_function"}
        assert valid_actions["<r,r>"] == {"<r,r> -> sample_function"}
        assert valid_actions["r"] == {"r -> [<e,r>, e]", "r -> [<d,r>, d]", "r -> [<r,r>, r]"}
        assert valid_actions["@START@"] == {"@START@ -> d", "@START@ -> r", "@START@ -> e"}

    def test_get_valid_actions_with_reverse(self):
        valid_actions = types.get_valid_actions(wt_types.COMMON_NAME_MAPPING, wt_types.COMMON_TYPE_SIGNATURE,
                                                wt_types.BASIC_TYPES)
        assert valid_actions['<d,e>'] == {'<d,e> -> [<<#1,#2>,<#2,#1>>, <e,d>]',
                                          '<d,e> -> fb:cell.cell.date',
                                          '<d,e> -> fb:cell.cell.number',
                                          '<d,e> -> fb:cell.cell.num2'
                                         }
