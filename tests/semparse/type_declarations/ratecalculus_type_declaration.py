"""
Defines all the types in the Linear Algebra domain.
"""

from overrides import overrides

from nltk.sem.logic import Type, BasicType, ANY_TYPE, ComplexType as NltkComplexType

from allennlp.data.semparse.type_declarations.type_declaration import ComplexType
from allennlp.data.semparse.type_declarations.type_declaration import NamedBasicType
from allennlp.data.semparse.type_declarations.type_declaration import UnaryOpType

BOOLEAN_TYPE = NamedBasicType("BOOLEAN")
DIMENSION_TYPE = NamedBasicType("DIMENSION")
OBJECT_TYPE = NamedBasicType("OBJECT")
NUMBER_TYPE = NamedBasicType("NUMBER")

BASIC_TYPES = {BOOLEAN_TYPE}
VALUE_TYPE = ComplexType(OBJECT_TYPE,
                         ComplexType(DIMENSION_TYPE, NUMBER_TYPE))
RATE_TYPE = ComplexType(OBJECT_TYPE,
                         ComplexType(DIMENSION_TYPE, ComplexType(DIMENSION_TYPE, NUMBER_TYPE)))

IDENTITY_TYPE = UnaryOpType()

# number
NUMBER_FUNCTION_TYPE = ComplexType(NUMBER_TYPE, NUMBER_TYPE)

# Unary numerical/boolean operations
UNARY_NUM_OP_TYPE = ComplexType(NUMBER_TYPE, NUMBER_TYPE)
UNARY_BOOL_OP_TYPE = ComplexType(BOOLEAN_TYPE, BOOLEAN_TYPE)

# Binary numerical/boolean operation: -
BINARY_NUM_OP_TYPE = ComplexType(NUMBER_TYPE, ComplexType(NUMBER_TYPE, NUMBER_TYPE))
BINARY_NUM_TO_BOOL_OP_TYPE = ComplexType(NUMBER_TYPE, ComplexType(NUMBER_TYPE, BOOLEAN_TYPE))
BINARY_BOOL_OP_TYPE = ComplexType(BOOLEAN_TYPE, ComplexType(BOOLEAN_TYPE, BOOLEAN_TYPE))

# and, or
CONJUNCTION_TYPE = BINARY_BOOL_OP_TYPE

COMMON_NAME_MAPPING = {"x": "X", "y": "Y", "p": "P", "q": "Q"}

COMMON_TYPE_SIGNATURE = {"X": OBJECT_TYPE, "Y": OBJECT_TYPE, "P": NUMBER_TYPE, "Q": NUMBER_TYPE}

def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature

add_common_name_with_type("Value", "V1", VALUE_TYPE)
add_common_name_with_type("Rate", "R1", RATE_TYPE)
add_common_name_with_type("And", "A", CONJUNCTION_TYPE)
add_common_name_with_type("Equals", "E", BINARY_NUM_TO_BOOL_OP_TYPE)
add_common_name_with_type("dollar", "D", DIMENSION_TYPE)
add_common_name_with_type("unit", "U", DIMENSION_TYPE)