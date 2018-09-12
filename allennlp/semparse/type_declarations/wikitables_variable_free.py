"""
Defines all types in the variable-free language for the WikiTablesQuestions dataset defined in the
following paper by Liang et al. (2018)
Memory Augmented Policy Optimization for Program Synthesis with Generalization.
"""


from allennlp.semparse.type_declarations.type_declaration import Type, NamedBasicType, ComplexType


# Basic types
# Note that there aren't cell types and part types in this gramamr. They are all of type string. The
# constants will still come from the `TableQuestionKnowledgeGraph`, and are identified as parts and
# cells, and the executor can process them accordingly, but the grammar does not differentiate
# between them. Also note that columns are basic types in this grammar.
ROW_TYPE = NamedBasicType("ROW")
# TODO(pradeep): Add different column types for string, number, and date columns, and may be a
# generic column type.
# "LCOLUMN" to ensure the signature will be "l", to avoid confusion with cell. I decided to omit the
# cell type eventually, and the decision to call this "LCOLUMN" is not relevant any more. But I'm
# not changing it for now, because I might end up adding the cell type back later.
COLUMN_TYPE = NamedBasicType("LCOLUMN")
NUMBER_TYPE = NamedBasicType("NUMBER")
DATE_TYPE = NamedBasicType("DATE")
STRING_TYPE = NamedBasicType("STRING")

BASIC_TYPES = {ROW_TYPE, COLUMN_TYPE, NUMBER_TYPE, DATE_TYPE, STRING_TYPE}
STARTING_TYPES = {NUMBER_TYPE, DATE_TYPE, STRING_TYPE}

# Complex types
# Type for selecting the value in a column in a set of rows. "select" and "mode" functions.
SELECT_TYPE = ComplexType(ROW_TYPE, ComplexType(COLUMN_TYPE, STRING_TYPE))

# Type for filtering rows given a column. "argmax", "argmin" and "same_as" (select all rows with the
# same value under the given column as the given row does under the given column)
# Note that the values used for comparison in "argmax" and "argmin" can only come from column
# lookups in this language. In LambdaDCS, there's a lambda function that is applied to the rows to
# get the values, but here, we simply have a column name.
ROW_FILTER_WITH_COLUMN = ComplexType(ROW_TYPE, ComplexType(COLUMN_TYPE, ROW_TYPE))

# "filter_number_greater", "filter_number_equals" etc.
ROW_FILTER_WITH_COLUMN_AND_NUMBER = ComplexType(ROW_TYPE,
                                                ComplexType(NUMBER_TYPE,
                                                            ComplexType(COLUMN_TYPE, ROW_TYPE)))

# "filter_date_greater", "filter_date_equals" etc.
ROW_FILTER_WITH_COLUMN_AND_DATE = ComplexType(ROW_TYPE,
                                              ComplexType(DATE_TYPE,
                                                          ComplexType(COLUMN_TYPE, ROW_TYPE)))

# "filter_in" and "filter_not_in"
ROW_FILTER_WITH_COLUMN_AND_STRING = ComplexType(ROW_TYPE,
                                                ComplexType(STRING_TYPE,
                                                            ComplexType(COLUMN_TYPE, ROW_TYPE)))

ROW_FILTER = ComplexType(ROW_TYPE, ROW_TYPE)  # first, last, previous, next etc.

# This language lets you count only rows!
COUNT_TYPE = ComplexType(ROW_TYPE, NUMBER_TYPE)

# Numerical operations on numbers in the given column. "max", "min", "sum", "average" etc.
ROW_NUM_OP = ComplexType(ROW_TYPE, ComplexType(COLUMN_TYPE, NUMBER_TYPE))

# Numerical difference within the given column.
NUM_DIFF_WITH_COLUMN = ComplexType(ROW_TYPE, ComplexType(ROW_TYPE, ComplexType(COLUMN_TYPE,
                                                                               NUMBER_TYPE)))

# Date function takes three numbers and makes a date
DATE_FUNCTION_TYPE = ComplexType(NUMBER_TYPE, ComplexType(NUMBER_TYPE, ComplexType(NUMBER_TYPE,
                                                                                   DATE_TYPE)))


COMMON_NAME_MAPPING = {}
COMMON_TYPE_SIGNATURE = {}


def add_common_name_with_type(name: str, mapping: str, type_signature: Type) -> None:
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature


add_common_name_with_type("all_rows", "R", ROW_TYPE)

# <r,<l,s>>
add_common_name_with_type("select", "S0", SELECT_TYPE)
add_common_name_with_type("mode", "S1", SELECT_TYPE)

# <r,<l,r>>
add_common_name_with_type("argmax", "F00", ROW_FILTER_WITH_COLUMN)
add_common_name_with_type("argmin", "F01", ROW_FILTER_WITH_COLUMN)
add_common_name_with_type("same_as", "F02", ROW_FILTER_WITH_COLUMN)

# <r,<n,<l,r>>>
add_common_name_with_type("filter_number_greater", "F10", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_greater_equals", "F11", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_lesser", "F12", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_lesser_equals", "F13", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_equals", "F14", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_not_equals", "F15", ROW_FILTER_WITH_COLUMN_AND_NUMBER)

# <r,<d,<l,r>>>
add_common_name_with_type("filter_date_greater", "F20", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_greater_equals", "F21", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_lesser", "F22", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_lesser_equals", "F23", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_equals", "F24", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_not_equals", "F25", ROW_FILTER_WITH_COLUMN_AND_DATE)

# <r,<s,<l,r>>>
add_common_name_with_type("filter_in", "F30", ROW_FILTER_WITH_COLUMN_AND_STRING)
add_common_name_with_type("filter_not_in", "F31", ROW_FILTER_WITH_COLUMN_AND_STRING)

# <r,r>
add_common_name_with_type("first", "R0", ROW_FILTER)
add_common_name_with_type("last", "R1", ROW_FILTER)
add_common_name_with_type("previous", "R2", ROW_FILTER)
add_common_name_with_type("next", "R3", ROW_FILTER)

# <r,n>
add_common_name_with_type("count", "C", COUNT_TYPE)

# <r,<l,n>>
add_common_name_with_type("max", "N0", ROW_NUM_OP)
add_common_name_with_type("min", "N1", ROW_NUM_OP)
add_common_name_with_type("average", "N2", ROW_NUM_OP)
add_common_name_with_type("sum", "N3", ROW_NUM_OP)

# <r,<r,<l,n>>>
add_common_name_with_type("diff", "D0", NUM_DIFF_WITH_COLUMN)

# <n,<n,<n,d>>>
add_common_name_with_type("date", "T0", DATE_FUNCTION_TYPE)
