# pylint: disable=no-self-use
from allennlp.data.dataset_readers.wikitables.type_declaration import ReverseType, IdentityType, ArgExtremeType
from allennlp.data.dataset_readers.wikitables.type_declaration import ConjunctionType, CountType, ComplexType
from allennlp.data.dataset_readers.wikitables.type_declaration import REVERSE_TYPE, IDENTITY_TYPE, CELL_TYPE
from allennlp.data.dataset_readers.wikitables.type_declaration import CONJUNCTION_TYPE, COUNT_TYPE, ROW_TYPE
from allennlp.data.dataset_readers.wikitables.type_declaration import ARG_EXTREME_TYPE, DATE_NUM_TYPE, ANY_TYPE
from allennlp.data.dataset_readers.wikitables.type_declaration import NamedBasicType

from allennlp.common.testing import AllenNlpTestCase


class TestPlaceholderTypeResolution(AllenNlpTestCase):
    def test_basic_types_conflict_on_names(self):
        type_a = NamedBasicType("A")
        type_b = NamedBasicType("B")
        assert type_a.resolve(type_b) is None

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

    def test_identity_type_resolves_correctly(self):
        # Resolution should fail against a basic type
        assert IDENTITY_TYPE.resolve(ROW_TYPE) is None

        # Resolution should fail against a complex type where the argument and return types are not same
        assert IDENTITY_TYPE.resolve(ComplexType(CELL_TYPE, ROW_TYPE)) is None

        # Resolution should resolve ANY_TYPE given the other type
        assert IDENTITY_TYPE.resolve(ComplexType(ANY_TYPE, ROW_TYPE)) == IdentityType(ROW_TYPE, ROW_TYPE)
        assert IDENTITY_TYPE.resolve(ComplexType(CELL_TYPE, ANY_TYPE)) == IdentityType(CELL_TYPE, CELL_TYPE)

        resolution = IDENTITY_TYPE.resolve(ComplexType(ComplexType(CELL_TYPE, ROW_TYPE),
                                                       ComplexType(CELL_TYPE, ROW_TYPE)))
        assert resolution == IdentityType(ComplexType(CELL_TYPE, ROW_TYPE), ComplexType(CELL_TYPE, ROW_TYPE))

    def test_conjunction_type_resolves_correctly(self):
        # Resolution must fail against a basic type and a complex type that returns a basic type
        assert CONJUNCTION_TYPE.resolve(CELL_TYPE) is None
        assert CONJUNCTION_TYPE.resolve(ComplexType(CELL_TYPE, ROW_TYPE)) is None

        # Resolution must fail against incompatible types
        assert CONJUNCTION_TYPE.resolve(ComplexType(ANY_TYPE, ComplexType(CELL_TYPE, ROW_TYPE))) is None
        assert CONJUNCTION_TYPE.resolve(ComplexType(ROW_TYPE, ComplexType(CELL_TYPE, ANY_TYPE))) is None
        assert CONJUNCTION_TYPE.resolve(ComplexType(ROW_TYPE, ComplexType(ANY_TYPE, CELL_TYPE))) is None

        # Resolution must resolve any types appropriately
        resolution = CONJUNCTION_TYPE.resolve(ComplexType(ROW_TYPE, ComplexType(ANY_TYPE, ROW_TYPE)))
        assert resolution == ConjunctionType(ROW_TYPE, ComplexType(ROW_TYPE, ROW_TYPE))

        resolution = CONJUNCTION_TYPE.resolve(ComplexType(ROW_TYPE, ComplexType(ANY_TYPE, ANY_TYPE)))
        assert resolution == ConjunctionType(ROW_TYPE, ComplexType(ROW_TYPE, ROW_TYPE))

        resolution = CONJUNCTION_TYPE.resolve(ComplexType(ANY_TYPE, ComplexType(ROW_TYPE, ANY_TYPE)))
        assert resolution == ConjunctionType(ROW_TYPE, ComplexType(ROW_TYPE, ROW_TYPE))

    def test_count_type_resolves_correctly(self):
        # Resolution should fail with basic type
        assert COUNT_TYPE.resolve(CELL_TYPE) is None

        # Resolution should fail when return type is not a number
        assert COUNT_TYPE.resolve(ComplexType(CELL_TYPE, ROW_TYPE)) is None

        # Resolution should resolve the return type to number
        assert COUNT_TYPE.resolve(ComplexType(CELL_TYPE, ANY_TYPE)) == CountType(CELL_TYPE, DATE_NUM_TYPE)
        assert COUNT_TYPE.resolve(ComplexType(ANY_TYPE, ANY_TYPE)) == CountType(ANY_TYPE, DATE_NUM_TYPE)

    def test_arg_extreme_type_resolves_correctly(self):
        # Resolution should fail on basic type
        assert ARG_EXTREME_TYPE.resolve(ROW_TYPE) is None

        assert ARG_EXTREME_TYPE.resolve(ComplexType(CELL_TYPE, ROW_TYPE)) is None

        resolution = ARG_EXTREME_TYPE.resolve(ComplexType(
                ANY_TYPE, ComplexType(ANY_TYPE,
                                      ComplexType(CELL_TYPE,
                                                  ComplexType(ComplexType(ANY_TYPE, CELL_TYPE), CELL_TYPE)))))
        assert resolution == ArgExtremeType(DATE_NUM_TYPE,
                                            ComplexType(DATE_NUM_TYPE,
                                                        ComplexType(DATE_NUM_TYPE,
                                                                    ComplexType(ComplexType(DATE_NUM_TYPE,
                                                                                            CELL_TYPE),
                                                                                CELL_TYPE))))
