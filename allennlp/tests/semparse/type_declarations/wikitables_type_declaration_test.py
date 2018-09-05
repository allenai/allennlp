# pylint: disable=no-self-use
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.type_declarations import type_declaration as base_types
from allennlp.semparse.type_declarations.type_declaration import (
        ComplexType,
        ANY_TYPE,
        )
from allennlp.semparse.type_declarations.wikitables_lambda_dcs import (
        ARG_EXTREME_TYPE,
        ArgExtremeType,
        CELL_TYPE,
        CONJUNCTION_TYPE,
        COUNT_TYPE,
        CountType,
        ROW_TYPE,
        REVERSE_TYPE,
        ReverseType,
        )


class WikiTablesTypeDeclarationTest(AllenNlpTestCase):
    def test_reverse_resolves_correctly(self):
        assert REVERSE_TYPE.resolve(CELL_TYPE) is None

        # Resolving against <?,<e,r>> should give <<r,e>,<e,r>>
        resolution = REVERSE_TYPE.resolve(ComplexType(ANY_TYPE, ComplexType(CELL_TYPE, ROW_TYPE)))
        assert resolution == ReverseType(ComplexType(ROW_TYPE, CELL_TYPE), ComplexType(CELL_TYPE, ROW_TYPE))

        # Resolving against <<r,?>,<e,?>> should give <<r,e>,<e,r>>
        resolution = REVERSE_TYPE.resolve(ComplexType(ComplexType(ROW_TYPE, ANY_TYPE),
                                                      ComplexType(CELL_TYPE, ANY_TYPE)))
        assert resolution == ReverseType(ComplexType(ROW_TYPE, CELL_TYPE), ComplexType(CELL_TYPE, ROW_TYPE))

        # Resolving against <<r,?>,?> should give <<r,?>,<?,r>>
        resolution = REVERSE_TYPE.resolve(ComplexType(ComplexType(ROW_TYPE, ANY_TYPE), ANY_TYPE))
        assert resolution == ReverseType(ComplexType(ROW_TYPE, ANY_TYPE), ComplexType(ANY_TYPE, ROW_TYPE))

        # Resolving against <<r,?>,<?,e>> should give None
        resolution = REVERSE_TYPE.resolve(ComplexType(ComplexType(ROW_TYPE, ANY_TYPE),
                                                      ComplexType(ANY_TYPE, CELL_TYPE)))
        assert resolution is None

    def test_conjunction_maps_to_correct_actions(self):
        valid_actions = base_types.get_valid_actions({'and': 'O'},
                                                     {'O': CONJUNCTION_TYPE},
                                                     {CELL_TYPE},
                                                     {CELL_TYPE})
        assert 'c -> [<#1,<#1,#1>>, c, c]' in valid_actions['c']

    def test_count_type_resolves_correctly(self):
        # Resolution should fail with basic type
        assert COUNT_TYPE.resolve(CELL_TYPE) is None

        # Resolution should fail when return type is not a number
        assert COUNT_TYPE.resolve(ComplexType(CELL_TYPE, ROW_TYPE)) is None

        # Resolution should resolve the return type to number
        assert COUNT_TYPE.resolve(ComplexType(CELL_TYPE, ANY_TYPE)) == CountType(CELL_TYPE)
        assert COUNT_TYPE.resolve(ComplexType(ANY_TYPE, ANY_TYPE)) == CountType(ANY_TYPE)

    def test_arg_extreme_type_resolves_correctly(self):
        # Resolution should fail on basic type
        assert ARG_EXTREME_TYPE.resolve(ROW_TYPE) is None

        assert ARG_EXTREME_TYPE.resolve(ComplexType(CELL_TYPE, ROW_TYPE)) is None

        other = ComplexType(ANY_TYPE,
                            ComplexType(ANY_TYPE,
                                        ComplexType(CELL_TYPE,
                                                    ComplexType(ComplexType(ANY_TYPE, CELL_TYPE),
                                                                CELL_TYPE))))
        resolution = ARG_EXTREME_TYPE.resolve(other)
        assert resolution == ArgExtremeType(CELL_TYPE)
