"""
Defines all the types in the Rate Calculus domain.
"""

from allennlp.semparse.type_declarations.type_declaration import ComplexType
from allennlp.semparse.type_declarations.type_declaration import NamedBasicType
from allennlp.semparse.type_declarations.type_declaration import UnaryOpType

BOOLEAN_TYPE = NamedBasicType("BOOLEAN")
DIMENSION_TYPE = NamedBasicType("DIMENSION")
OBJECT_TYPE = NamedBasicType("OBJECT")
NUMBER_TYPE = NamedBasicType("NUMBER")

BASIC_TYPES = {BOOLEAN_TYPE, NUMBER_TYPE, DIMENSION_TYPE, OBJECT_TYPE}
VALUE_TYPE = ComplexType(OBJECT_TYPE,
                         ComplexType(DIMENSION_TYPE, NUMBER_TYPE))
RATE_TYPE = ComplexType(OBJECT_TYPE,
                        ComplexType(DIMENSION_TYPE,
                                    ComplexType(DIMENSION_TYPE, NUMBER_TYPE)))
ISPART_TYPE = ComplexType(OBJECT_TYPE,
                          ComplexType(OBJECT_TYPE, BOOLEAN_TYPE))

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

COMMON_NAME_MAPPING = {}
COMMON_TYPE_SIGNATURE = {}

def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature

# Rate Calculus Operators
add_common_name_with_type("Value", "V", VALUE_TYPE)
add_common_name_with_type("Rate", "R", RATE_TYPE)
add_common_name_with_type("IsPart", "J", ISPART_TYPE)

# Linear Algebra Operators
add_common_name_with_type("And", "A", CONJUNCTION_TYPE)
add_common_name_with_type("Equals", "E", BINARY_NUM_TO_BOOL_OP_TYPE)
add_common_name_with_type("Plus", "P", BINARY_NUM_OP_TYPE)
add_common_name_with_type("Minus", "M", BINARY_NUM_OP_TYPE)
add_common_name_with_type("Times", "T", BINARY_NUM_OP_TYPE)
add_common_name_with_type("Div", "D", BINARY_NUM_OP_TYPE)

# Object Dimensions
add_common_name_with_type("Unit0", "U0", DIMENSION_TYPE)
add_common_name_with_type("Unit1", "U1", DIMENSION_TYPE)
add_common_name_with_type("Dollar", "U5", DIMENSION_TYPE)
add_common_name_with_type("Unit", "U6", DIMENSION_TYPE)

# Object Variables
add_common_name_with_type("o0", "O0", OBJECT_TYPE)
add_common_name_with_type("o1", "O1", OBJECT_TYPE)
add_common_name_with_type("o2", "O2", OBJECT_TYPE)
add_common_name_with_type("o3", "O3", OBJECT_TYPE)
add_common_name_with_type("o4", "O4", OBJECT_TYPE)
add_common_name_with_type("o5", "O5", OBJECT_TYPE)

# Numerical Variables
add_common_name_with_type("x0", "X0", NUMBER_TYPE)
add_common_name_with_type("x1", "X1", NUMBER_TYPE)
# Numerical Query Variables
add_common_name_with_type("q0", "Q0", NUMBER_TYPE)
add_common_name_with_type("q1", "Q1", NUMBER_TYPE)
