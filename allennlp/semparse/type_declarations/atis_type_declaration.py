from typing import List, Set
from overrides import overrides

from nltk.sem.logic import TRUTH_TYPE, BasicType, EntityType, Type

from allennlp.semparse.type_declarations.type_declaration import ComplexType, HigherOrderType, NamedBasicType

# Basic Types
NUM_TYPE = NamedBasicType("NUM")
STRING_TYPE = NamedBasicType("STRING")

FROM_BASIC_TYPE = NamedBasicType("FROM")
WHERE_BASIC_TYPE = NamedBasicType("WHERE")

# Complex Types
CONJ_TYPE = ComplexType(TRUTH_TYPE, ComplexType(TRUTH_TYPE, TRUTH_TYPE))
# TODO Add other Binops
BINOP_TYPE = ComplexType(NUM_TYPE, ComplexType(NUM_TYPE, TRUTH_TYPE))

SELECT_TYPE = ComplexType(STRING_TYPE, ComplexType(FROM_BASIC_TYPE, ComplexType(WHERE_BASIC_TYPE, NUM_TYPE))) 
FROM_TYPE = ComplexType(STRING_TYPE, FROM_BASIC_TYPE)
WHERE_TYPE = ComplexType(TRUTH_TYPE, WHERE_BASIC_TYPE)

COMMON_NAME_MAPPING = {}
COMMON_TYPE_SIGNATURE = {}

def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature

add_common_name_with_type("SELECT", "S0", SELECT_TYPE)
add_common_name_with_type("FROM", "F0", FROM_TYPE)
add_common_name_with_type("WHERE", "W0", WHERE_TYPE)

add_common_name_with_type("AND", "C0", CONJ_TYPE)
add_common_name_with_type("OR", "C1", CONJ_TYPE)

add_common_name_with_type(">=", "B0", BINOP_TYPE)





