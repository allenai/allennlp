"""
Defines all the types in the WikitablesQuestions domain.
"""
from typing import List, Optional, Set
from overrides import overrides

from nltk.sem.logic import Type, BasicType, EntityType, ANY_TYPE

from allennlp.data.semparse.type_declarations.type_declaration import ComplexType, HigherOrderType, IdentityType
from allennlp.data.semparse.type_declarations.type_declaration import PlaceholderType, NamedBasicType


class ReverseType(PlaceholderType, HigherOrderType):
    """
    ReverseType is a kind of ``PlaceholderType`` where type resolution involves matching the return
    type with the reverse of the argument type. So all we care about are the types of the surrounding
    expressions, and return a resolution that matches whatever parts are present in the type signatures
    of the arguments and the return expressions.

    Following are the resolutions for some example type signatures being matched against::

        <?, <e,r>>      :   <<r,e>, <e,r>>
        <<r,?>, <e,?>>  :   <<r,e>, <e,r>>
        <<r,?>, ?>      :   <<r,?>, <?,r>>>
        <<r,?>, <?,e>>  :   None
    """
    def __init__(self, first: Type, second: Type) -> None:
        super().__init__(num_arguments=1, first=first, second=second)

    @property
    def _signature(self) -> str:
        return "<<#1,#2>,<#2,#1>>"

    @overrides
    def resolve(self, other: Type) -> Optional[Type]:
        # Idea: Since its signature is <<#1,#2>,<#2,#1>> no information about types in self is relevant.
        # All that matters is that other.fiirst resolves against the reverse of other.second and vice versa.
        if not isinstance(other, ComplexType):
            return None
        # other.first and other.second are the argument and return types respectively.
        reversed_second = ComplexType(other.second.second, other.second.first)
        other_first = other.first.resolve(reversed_second)
        if not other_first:
            return None
        reversed_first = ComplexType(other_first.second, other_first.first)
        other_second = other.second.resolve(reversed_first)
        if not other_second:
            return None
        return ReverseType(other_first, other_second)

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        return ComplexType(argument_type.second, argument_type.first)

    @overrides
    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        basic_first_types = basic_types if self.first.first == ANY_TYPE else {self.first.first}
        basic_second_types = basic_types if self.first.second == ANY_TYPE else {self.first.second}
        substitutions = []
        for first_type in basic_first_types:
            for second_type in basic_second_types:
                substituted_first = ComplexType(first_type, second_type)
                substituted_second = ComplexType(second_type, first_type)
                substitutions.append(ReverseType(substituted_first, substituted_second))
        return substitutions


class ConjunctionType(PlaceholderType):
    """
    ``ConjunctionType`` takes an entity of any type and returns a function that takes and returns the same
    type. That is, its signature is <#1, <#1, #1>>
    """
    @property
    def _signature(self) -> str:
        return "<#1,<#1,#1>>"

    @overrides
    def resolve(self, other: Type) -> Optional[Type]:
        """See ``PlaceholderType.resolve``"""
        if not isinstance(other, ComplexType):
            return None
        if not isinstance(other.second, ComplexType):
            return None
        other_first = other.first.resolve(other.second.first)
        if other_first is None:
            return None
        other_first = other_first.resolve(other.second.second)
        if not other_first:
            return None
        other_second = other.second.resolve(ComplexType(other_first, other_first))
        if not other_second:
            return None
        return ConjunctionType(other_first, other_second)

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        return ComplexType(argument_type, argument_type)

    @overrides
    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        if self.first != ANY_TYPE:
            return [self]
        return [ConjunctionType(basic_type, basic_type) for basic_type in basic_types]


class ArgExtremeType(PlaceholderType):
    """
    This is the type for argmax and argmin in Sempre. The type signature is <d,<d,<#1,<<d,#1>,#1>>>>.
    Example: (argmax (number 1) (number 1) (fb:row.row.league fb:cell.usl_a_league) fb:row.row.index)
    meaning, of the subset of rows where league == usl_a_league, find the row with the maximum index.
    """
    @property
    def _signature(self) -> str:
        return "<d,<d,<#1,<<d,#1>,#1>>>>"

    @overrides
    def resolve(self, other: Type) -> Optional[Type]:
        """See ``PlaceholderType.resolve``"""
        if not isinstance(other, ComplexType):
            return None
        expected_second = ComplexType(DATE_NUM_TYPE,
                                      ComplexType(ANY_TYPE, ComplexType(ComplexType(DATE_NUM_TYPE, ANY_TYPE),
                                                                        ANY_TYPE)))
        resolved_second = other.second.resolve(expected_second)
        if resolved_second is None:
            return None

        try:
            # This is the first #1 in the type signature above.
            selector_function_type = resolved_second.second.first
            # This is the second #1 in the type signature above.
            quant_function_argument_type = resolved_second.second.second.first.second
            # This is the third #1 in the type signature above.
            return_type = resolved_second.second.second.second

            # All three placeholder (ph) types above should resolve against each other.
            resolved_first_ph = selector_function_type.resolve(quant_function_argument_type)
            resolved_first_ph.resolve(return_type)

            resolved_second_ph = quant_function_argument_type.resolve(resolved_first_ph)
            resolved_second_ph.resolve(return_type)

            resolved_third_ph = return_type.resolve(resolved_first_ph)
            resolved_third_ph = return_type.resolve(resolved_second_ph)

            if not resolved_first_ph or not resolved_second_ph or not resolved_third_ph:
                return None

            return ArgExtremeType(DATE_NUM_TYPE,
                                  ComplexType(DATE_NUM_TYPE,
                                              ComplexType(resolved_first_ph,
                                                          ComplexType(ComplexType(DATE_NUM_TYPE,
                                                                                  resolved_second_ph),
                                                                      resolved_third_ph))))
        except AttributeError:
            return None

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        # Returning <d,<#1,<<d,#1>,#1>>>.
        # This is called after the placeholders are resolved.
        return self.second

    @overrides
    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        if self.second.second.first != ANY_TYPE:
            return [self]
        return [self._substitute_basic_type(basic_type) for basic_type in basic_types]

    @classmethod
    def _substitute_basic_type(cls, basic_type: BasicType) -> 'ArgExtremeType':
        return cls(DATE_NUM_TYPE,
                   ComplexType(DATE_NUM_TYPE,
                               ComplexType(basic_type,
                                           ComplexType(ComplexType(DATE_NUM_TYPE, basic_type),
                                                       basic_type))))



class CountType(PlaceholderType):
    """
    Type of a function that counts arbitrary things. Signature is <#1,d>.
    """
    @property
    def _signature(self) -> str:
        return "<#1,d>"

    @overrides
    def resolve(self, other: Type) -> Type:
        """See ``PlaceholderType.resolve``"""
        if not isinstance(other, ComplexType):
            return None
        resolved_second = DATE_NUM_TYPE.resolve(other.second)
        if not resolved_second:
            return None
        return CountType(ANY_TYPE, resolved_second)

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        return DATE_NUM_TYPE

    @overrides
    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        if self.first != ANY_TYPE:
            return [self]
        return [CountType(basic_type, DATE_NUM_TYPE) for basic_type in basic_types]


CELL_TYPE = EntityType()
PART_TYPE = NamedBasicType("PART")
ROW_TYPE = NamedBasicType("ROW")
# TODO (pradeep): Merging dates and nums. Can define a hierarchy instead.
DATE_NUM_TYPE = NamedBasicType("DATENUM")

BASIC_TYPES = {CELL_TYPE, PART_TYPE, ROW_TYPE, DATE_NUM_TYPE}
# Functions like fb:row.row.year.
COLUMN_TYPE = ComplexType(CELL_TYPE, ROW_TYPE)
# fb:cell.cell.part
PART2CELL_TYPE = ComplexType(PART_TYPE, CELL_TYPE)
# fb:cell.cell.date
CELL2DATE_NUM_TYPE = ComplexType(DATE_NUM_TYPE, CELL_TYPE)
# number
NUMBER_FUNCTION_TYPE = ComplexType(EntityType(), DATE_NUM_TYPE)
# date (Signature: <e,<e,<e,d>>>; Example: (date 1982 -1 -1))
DATE_FUNCTION_TYPE = ComplexType(EntityType(),
                                 ComplexType(EntityType(), ComplexType(EntityType(), DATE_NUM_TYPE)))
# Unary numerical operations: max, min, >, <, sum etc.
UNARY_NUM_OP_TYPE = ComplexType(DATE_NUM_TYPE, DATE_NUM_TYPE)
# Binary numerical operation: -
BINARY_NUM_OP_TYPE = ComplexType(DATE_NUM_TYPE, ComplexType(DATE_NUM_TYPE, DATE_NUM_TYPE))
# next
NEXT_ROW_TYPE = ComplexType(ROW_TYPE, ROW_TYPE)
# reverse
REVERSE_TYPE = ReverseType(ComplexType(ANY_TYPE, ANY_TYPE), ComplexType(ANY_TYPE, ANY_TYPE))
# !=, fb:type.object.type
# fb:type.object.type takes a type and returns all objects of that type.
IDENTITY_TYPE = IdentityType(ANY_TYPE, ANY_TYPE)
# index
ROW_INDEX_TYPE = ComplexType(DATE_NUM_TYPE, ROW_TYPE)
# count
COUNT_TYPE = CountType(ANY_TYPE, DATE_NUM_TYPE)
# and, or
CONJUNCTION_TYPE = ConjunctionType(ANY_TYPE, ANY_TYPE)
# argmax, argmin
ARG_EXTREME_TYPE = ArgExtremeType(DATE_NUM_TYPE,
                                  ComplexType(DATE_NUM_TYPE,
                                              ComplexType(ANY_TYPE,
                                                          ComplexType(ComplexType(DATE_NUM_TYPE,
                                                                                  ANY_TYPE),
                                                                      ANY_TYPE))))



COMMON_NAME_MAPPING = {"lambda": "\\", "var": "V", "x": "X"}


COMMON_TYPE_SIGNATURE = {"V": IDENTITY_TYPE, "X": ANY_TYPE}


def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature

add_common_name_with_type("reverse", "R", REVERSE_TYPE)
add_common_name_with_type("argmax", "A0", ARG_EXTREME_TYPE)
add_common_name_with_type("argmin", "A1", ARG_EXTREME_TYPE)
add_common_name_with_type("max", "M0", UNARY_NUM_OP_TYPE)
add_common_name_with_type("min", "M1", UNARY_NUM_OP_TYPE)
add_common_name_with_type("and", "A", CONJUNCTION_TYPE)
add_common_name_with_type("or", "O", CONJUNCTION_TYPE)
add_common_name_with_type("fb:row.row.next", "N", NEXT_ROW_TYPE)
add_common_name_with_type("number", "I", NUMBER_FUNCTION_TYPE)
add_common_name_with_type("date", "D0", DATE_FUNCTION_TYPE)
add_common_name_with_type("fb:cell.cell.part", "P", PART2CELL_TYPE)
add_common_name_with_type("fb:cell.cell.date", "D1", CELL2DATE_NUM_TYPE)
add_common_name_with_type("fb:cell.cell.number", "I1", CELL2DATE_NUM_TYPE)
add_common_name_with_type("fb:cell.cell.num2", "I2", CELL2DATE_NUM_TYPE)
add_common_name_with_type("fb:row.row.index", "W", ROW_INDEX_TYPE)
add_common_name_with_type("fb:type.row", "T0", ROW_TYPE)
add_common_name_with_type("fb:type.object.type", "T", IDENTITY_TYPE)
add_common_name_with_type("count", "C", COUNT_TYPE)
add_common_name_with_type("!=", "Q", IDENTITY_TYPE)
add_common_name_with_type(">", "G0", UNARY_NUM_OP_TYPE)
add_common_name_with_type(">=", "G1", UNARY_NUM_OP_TYPE)
add_common_name_with_type("<", "L0", UNARY_NUM_OP_TYPE)
add_common_name_with_type("<=", "L1", UNARY_NUM_OP_TYPE)
add_common_name_with_type("sum", "S0", UNARY_NUM_OP_TYPE)
add_common_name_with_type("avg", "S1", UNARY_NUM_OP_TYPE)
add_common_name_with_type("-", "F", BINARY_NUM_OP_TYPE)  # subtraction
