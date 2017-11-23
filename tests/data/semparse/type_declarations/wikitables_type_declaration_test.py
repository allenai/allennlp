# pylint: disable=no-self-use
from allennlp.data.semparse.type_declarations import wikitables_type_declaration as types

from allennlp.common.testing import AllenNlpTestCase


class TestPlaceholderTypeResolution(AllenNlpTestCase):
    def test_basic_types_conflict_on_names(self):
        type_a = types.NamedBasicType("A")
        type_b = types.NamedBasicType("B")
        assert type_a.resolve(type_b) is None

    def test_reverse_resolves_correctly(self):
        assert types.REVERSE_TYPE.resolve(types.CELL_TYPE) is None

        # Resolving against <?,<e,r>> should give <<r,e>,<e,r>>
        resolution = types.REVERSE_TYPE.resolve(types.ComplexType(types.ANY_TYPE,
                                                                  types.ComplexType(types.CELL_TYPE,
                                                                                    types.ROW_TYPE)))
        assert resolution == types.ReverseType(types.ComplexType(types.ROW_TYPE, types.CELL_TYPE),
                                               types.ComplexType(types.CELL_TYPE, types.ROW_TYPE))

        # Resolving against <<r,?>,<e,?>> should give <<r,e>,<e,r>>
        resolution = types.REVERSE_TYPE.resolve(types.ComplexType(types.ComplexType(types.ROW_TYPE,
                                                                                    types.ANY_TYPE),
                                                                  types.ComplexType(types.CELL_TYPE,
                                                                                    types.ANY_TYPE)))
        assert resolution == types.ReverseType(types.ComplexType(types.ROW_TYPE, types.CELL_TYPE),
                                               types.ComplexType(types.CELL_TYPE, types.ROW_TYPE))

        # Resolving against <<r,?>,?> should give <<r,?>,<?,r>>
        resolution = types.REVERSE_TYPE.resolve(types.ComplexType(types.ComplexType(types.ROW_TYPE,
                                                                                    types.ANY_TYPE),
                                                                  types.ANY_TYPE))
        assert resolution == types.ReverseType(types.ComplexType(types.ROW_TYPE, types.ANY_TYPE),
                                               types.ComplexType(types.ANY_TYPE, types.ROW_TYPE))

        # Resolving against <<r,?>,<?,e>> should give None
        resolution = types.REVERSE_TYPE.resolve(types.ComplexType(types.ComplexType(types.ROW_TYPE,
                                                                                    types.ANY_TYPE),
                                                                  types.ComplexType(types.ANY_TYPE,
                                                                                    types.CELL_TYPE)))
        assert resolution is None

    def test_identity_type_resolves_correctly(self):
        # Resolution should fail against a basic type
        assert types.IDENTITY_TYPE.resolve(types.ROW_TYPE) is None

        # Resolution should fail against a complex type where the argument and return types are not same
        assert types.IDENTITY_TYPE.resolve(types.ComplexType(types.CELL_TYPE, types.ROW_TYPE)) is None

        # Resolution should resolve ANY_TYPE given the other type
        assert types.IDENTITY_TYPE.resolve(types.ComplexType(types.ANY_TYPE, types.ROW_TYPE)) == \
        types.IdentityType(types.ROW_TYPE, types.ROW_TYPE)
        assert types.IDENTITY_TYPE.resolve(types.ComplexType(types.CELL_TYPE, types.ANY_TYPE)) == \
        types.IdentityType(types.CELL_TYPE, types.CELL_TYPE)

        resolution = types.IDENTITY_TYPE.resolve(types.ComplexType(types.ComplexType(types.CELL_TYPE,
                                                                                     types.ROW_TYPE),
                                                                   types.ComplexType(types.CELL_TYPE,
                                                                                     types.ROW_TYPE)))
        assert resolution == types.IdentityType(types.ComplexType(types.CELL_TYPE,
                                                                  types.ROW_TYPE),
                                                types.ComplexType(types.CELL_TYPE, types.ROW_TYPE))

    def test_conjunction_type_resolves_correctly(self):
        # Resolution must fail against a basic type and a complex type that returns a basic type
        assert types.CONJUNCTION_TYPE.resolve(types.CELL_TYPE) is None
        assert types.CONJUNCTION_TYPE.resolve(types.ComplexType(types.CELL_TYPE, types.ROW_TYPE)) is None

        # Resolution must fail against incompatible types
        assert types.CONJUNCTION_TYPE.resolve(types.ComplexType(types.ANY_TYPE,
                                                                types.ComplexType(types.CELL_TYPE,
                                                                                  types.ROW_TYPE))) is None
        assert types.CONJUNCTION_TYPE.resolve(types.ComplexType(types.ROW_TYPE,
                                                                types.ComplexType(types.CELL_TYPE,
                                                                                  types.ANY_TYPE))) is None
        assert types.CONJUNCTION_TYPE.resolve(types.ComplexType(types.ROW_TYPE,
                                                                types.ComplexType(types.ANY_TYPE,
                                                                                  types.CELL_TYPE))) is None

        # Resolution must resolve any types appropriately
        resolution = types.CONJUNCTION_TYPE.resolve(types.ComplexType(types.ROW_TYPE,
                                                                      types.ComplexType(types.ANY_TYPE,
                                                                                        types.ROW_TYPE)))
        assert resolution == types.ConjunctionType(types.ROW_TYPE, types.ComplexType(types.ROW_TYPE,
                                                                                     types.ROW_TYPE))

        resolution = types.CONJUNCTION_TYPE.resolve(types.ComplexType(types.ROW_TYPE,
                                                                      types.ComplexType(types.ANY_TYPE,
                                                                                        types.ANY_TYPE)))
        assert resolution == types.ConjunctionType(types.ROW_TYPE, types.ComplexType(types.ROW_TYPE,
                                                                                     types.ROW_TYPE))

        resolution = types.CONJUNCTION_TYPE.resolve(types.ComplexType(types.ANY_TYPE,
                                                                      types.ComplexType(types.ROW_TYPE,
                                                                                        types.ANY_TYPE)))
        assert resolution == types.ConjunctionType(types.ROW_TYPE, types.ComplexType(types.ROW_TYPE,
                                                                                     types.ROW_TYPE))

    def test_count_type_resolves_correctly(self):
        # Resolution should fail with basic type
        assert types.COUNT_TYPE.resolve(types.CELL_TYPE) is None

        # Resolution should fail when return type is not a number
        assert types.COUNT_TYPE.resolve(types.ComplexType(types.CELL_TYPE, types.ROW_TYPE)) is None

        # Resolution should resolve the return type to number
        assert types.COUNT_TYPE.resolve(types.ComplexType(types.CELL_TYPE, types.ANY_TYPE)) == \
                types.CountType(types.CELL_TYPE, types.DATE_NUM_TYPE)
        assert types.COUNT_TYPE.resolve(types.ComplexType(types.ANY_TYPE, types.ANY_TYPE)) == \
                types.CountType(types.ANY_TYPE, types.DATE_NUM_TYPE)

    def test_arg_extreme_type_resolves_correctly(self):
        # Resolution should fail on basic type
        assert types.ARG_EXTREME_TYPE.resolve(types.ROW_TYPE) is None

        assert types.ARG_EXTREME_TYPE.resolve(types.ComplexType(types.CELL_TYPE, types.ROW_TYPE)) is None

        other = types.ComplexType(types.ANY_TYPE, types.ComplexType(types.ANY_TYPE, \
                types.ComplexType(types.CELL_TYPE, types.ComplexType(types.ComplexType(types.ANY_TYPE,
                                                                                       types.CELL_TYPE),
                                                                     types.CELL_TYPE))))
        resolution = types.ARG_EXTREME_TYPE.resolve(other)
        assert resolution == types.ArgExtremeType(types.DATE_NUM_TYPE,\
                             types.ComplexType(types.DATE_NUM_TYPE, types.ComplexType(types.DATE_NUM_TYPE,\
                             types.ComplexType(types.ComplexType(types.DATE_NUM_TYPE, types.CELL_TYPE),\
                             types.CELL_TYPE))))
