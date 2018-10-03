"""
Defines all types in the variable-free language for the WikiTablesQuestions dataset defined in the
following paper by Liang et al. (2018)
Memory Augmented Policy Optimization for Program Synthesis with Generalization.
"""


from allennlp.semparse.type_declarations.type_declaration import (Type, NamedBasicType, ComplexType,
                                                                  MultiMatchNamedBasicType)


# Basic types
# Note that there aren't cell types and part types in this gramamr. They are all of type string. The
# constants will still come from the `TableQuestionKnowledgeGraph`, and are identified as parts and
# cells, and the executor can process them accordingly, but the grammar does not differentiate
# between them. Also note that columns are basic types in this grammar.
ROW_TYPE = NamedBasicType("ROW")
# The following three type signatures are assigned such that they're easy to understand from the
# first letters, while being different from the cell types.
DATE_COLUMN_TYPE = NamedBasicType("YCOLUMN")  # Y for year
NUMBER_COLUMN_TYPE = NamedBasicType("FCOLUMN")  # F for float
STRING_COLUMN_TYPE = NamedBasicType("TCOLUMN")  # T for token
GENERIC_COLUMN_TYPE = MultiMatchNamedBasicType("GCOLUMN", [STRING_COLUMN_TYPE, DATE_COLUMN_TYPE,
                                                           NUMBER_COLUMN_TYPE])
COMPARABLE_COLUMN_TYPE = MultiMatchNamedBasicType("CCOLUMN", [NUMBER_COLUMN_TYPE, DATE_COLUMN_TYPE])

NUMBER_TYPE = NamedBasicType("NUMBER")
DATE_TYPE = NamedBasicType("DATE")
STRING_TYPE = NamedBasicType("STRING")

BASIC_TYPES = {ROW_TYPE, GENERIC_COLUMN_TYPE, COMPARABLE_COLUMN_TYPE, NUMBER_COLUMN_TYPE,
               STRING_COLUMN_TYPE, DATE_COLUMN_TYPE, NUMBER_TYPE, DATE_TYPE, STRING_TYPE}
STARTING_TYPES = {NUMBER_TYPE, DATE_TYPE, STRING_TYPE}

# Complex types
# Type for selecting the value in a column in a set of rows. "select" and "mode" functions.
SELECT_TYPE = ComplexType(ROW_TYPE, ComplexType(GENERIC_COLUMN_TYPE, STRING_TYPE))

# Type for filtering rows given a column. "argmax", "argmin" and "same_as" (select all rows with the
# same value under the given column as the given row does under the given column). While "same_as"
# takes any column, "argmax" and "argmin" take only comparable columns (i.e. dates or numbers).
# Note that the values used for comparison in "argmax" and "argmin" can only come from column
# lookups in this language. In LambdaDCS, there's a lambda function that is applied to the rows to
# get the values, but here, we simply have a column name.
ROW_FILTER_WITH_GENERIC_COLUMN = ComplexType(ROW_TYPE, ComplexType(GENERIC_COLUMN_TYPE, ROW_TYPE))
ROW_FILTER_WITH_COMPARABLE_COLUMN = ComplexType(ROW_TYPE, ComplexType(COMPARABLE_COLUMN_TYPE, ROW_TYPE))

# "filter_number_greater", "filter_number_equals" etc.
ROW_FILTER_WITH_COLUMN_AND_NUMBER = ComplexType(ROW_TYPE,
                                                ComplexType(NUMBER_COLUMN_TYPE,
                                                            ComplexType(NUMBER_TYPE, ROW_TYPE)))

# "filter_date_greater", "filter_date_equals" etc.
ROW_FILTER_WITH_COLUMN_AND_DATE = ComplexType(ROW_TYPE,
                                              ComplexType(DATE_COLUMN_TYPE,
                                                          ComplexType(DATE_TYPE, ROW_TYPE)))

# "filter_in" and "filter_not_in"
ROW_FILTER_WITH_COLUMN_AND_STRING = ComplexType(ROW_TYPE,
                                                ComplexType(STRING_COLUMN_TYPE,
                                                            ComplexType(STRING_TYPE, ROW_TYPE)))

ROW_FILTER = ComplexType(ROW_TYPE, ROW_TYPE)  # first, last, previous, next etc.

# This language lets you count only rows!
COUNT_TYPE = ComplexType(ROW_TYPE, NUMBER_TYPE)

# Numerical operations on numbers in the given column. "max", "min", "sum", "average" etc.
ROW_NUM_OP = ComplexType(ROW_TYPE, ComplexType(NUMBER_COLUMN_TYPE, NUMBER_TYPE))

# Numerical difference within the given column.
NUM_DIFF_WITH_COLUMN = ComplexType(ROW_TYPE, ComplexType(ROW_TYPE, ComplexType(NUMBER_COLUMN_TYPE,
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

# <r,<g,s>>
add_common_name_with_type("select", "S0", SELECT_TYPE)
add_common_name_with_type("mode", "S1", SELECT_TYPE)

# <r,<c,r>>
add_common_name_with_type("argmax", "F00", ROW_FILTER_WITH_COMPARABLE_COLUMN)
add_common_name_with_type("argmin", "F01", ROW_FILTER_WITH_COMPARABLE_COLUMN)

# <r,<g,r>>
add_common_name_with_type("same_as", "F02", ROW_FILTER_WITH_GENERIC_COLUMN)

# <r,<f,<n,r>>>
add_common_name_with_type("filter_number_greater", "F10", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_greater_equals", "F11", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_lesser", "F12", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_lesser_equals", "F13", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_equals", "F14", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
add_common_name_with_type("filter_number_not_equals", "F15", ROW_FILTER_WITH_COLUMN_AND_NUMBER)

# <r,<y,<d,r>>>
add_common_name_with_type("filter_date_greater", "F20", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_greater_equals", "F21", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_lesser", "F22", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_lesser_equals", "F23", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_equals", "F24", ROW_FILTER_WITH_COLUMN_AND_DATE)
add_common_name_with_type("filter_date_not_equals", "F25", ROW_FILTER_WITH_COLUMN_AND_DATE)

# <r,<t,<s,r>>>
add_common_name_with_type("filter_in", "F30", ROW_FILTER_WITH_COLUMN_AND_STRING)
add_common_name_with_type("filter_not_in", "F31", ROW_FILTER_WITH_COLUMN_AND_STRING)

# <r,r>
add_common_name_with_type("first", "R0", ROW_FILTER)
add_common_name_with_type("last", "R1", ROW_FILTER)
add_common_name_with_type("previous", "R2", ROW_FILTER)
add_common_name_with_type("next", "R3", ROW_FILTER)

# <r,n>
add_common_name_with_type("count", "C", COUNT_TYPE)

# <r,<f,n>>
add_common_name_with_type("max", "N0", ROW_NUM_OP)
add_common_name_with_type("min", "N1", ROW_NUM_OP)
add_common_name_with_type("average", "N2", ROW_NUM_OP)
add_common_name_with_type("sum", "N3", ROW_NUM_OP)

# <r,<r,<f,n>>>
add_common_name_with_type("diff", "D0", NUM_DIFF_WITH_COLUMN)

# <n,<n,<n,d>>>
add_common_name_with_type("date", "T0", DATE_FUNCTION_TYPE)
