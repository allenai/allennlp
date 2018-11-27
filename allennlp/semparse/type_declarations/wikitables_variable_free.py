"""
Defines all types in the variable-free language for the WikiTablesQuestions dataset defined in the
following paper by Liang et al. (2018)
Memory Augmented Policy Optimization for Program Synthesis with Generalization.
"""


from allennlp.semparse.type_declarations.type_declaration import (NamedBasicType, ComplexType,
                                                                  MultiMatchNamedBasicType,
                                                                  NameMapper)


# Basic types
# Note that there aren't cell types and part types in this gramamr. They are all of type string. The
# constants will still come from the `TableQuestionKnowledgeGraph`, and are identified as parts and
# cells, and the executor can process them accordingly, but the grammar does not differentiate
# between them. Also note that columns are basic types in this grammar.
ROW_TYPE = NamedBasicType("ROW")
# The following three type signatures are assigned such that they're easy to understand from the
# first letters, while being different from the cell types.
DATE_COLUMN_TYPE = NamedBasicType("MCOLUMN")  # M for month
NUMBER_COLUMN_TYPE = NamedBasicType("FCOLUMN")  # F for float
STRING_COLUMN_TYPE = NamedBasicType("TCOLUMN")  # T for token
GENERIC_COLUMN_TYPE = MultiMatchNamedBasicType("GCOLUMN", [STRING_COLUMN_TYPE, DATE_COLUMN_TYPE,
                                                           NUMBER_COLUMN_TYPE])
COMPARABLE_COLUMN_TYPE = MultiMatchNamedBasicType("CCOLUMN", [NUMBER_COLUMN_TYPE, DATE_COLUMN_TYPE])

NUMBER_TYPE = NamedBasicType("NUMBER")
DATE_TYPE = NamedBasicType("DATE")
STRING_TYPE = NamedBasicType("STRING")

# We start with just the generic column type, and add the specific column types to the set only if
# we see the corresponding types in the table.
BASIC_TYPES = {ROW_TYPE, GENERIC_COLUMN_TYPE, NUMBER_TYPE, DATE_TYPE, STRING_TYPE}
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


# We have one name mapper for all generic functions, and different ones for each of the column types
# because we will add those to the final name mapping only if needed, based on the table content.
generic_name_mapper = NameMapper()  # pylint: disable=invalid-name
string_column_name_mapper = NameMapper(alias_prefix="S")  # pylint: disable=invalid-name
number_column_name_mapper = NameMapper(alias_prefix="N")  # pylint: disable=invalid-name
date_column_name_mapper = NameMapper(alias_prefix="D")  # pylint: disable=invalid-name
comparable_column_name_mapper = NameMapper(alias_prefix="O")  # pylint: disable=invalid-name

generic_name_mapper.map_name_with_signature("all_rows", ROW_TYPE)

# <r,<g,s>>
generic_name_mapper.map_name_with_signature("select", SELECT_TYPE)
generic_name_mapper.map_name_with_signature("mode", SELECT_TYPE)

# <r,<c,r>>
comparable_column_name_mapper.map_name_with_signature("argmax", ROW_FILTER_WITH_COMPARABLE_COLUMN)
comparable_column_name_mapper.map_name_with_signature("argmin", ROW_FILTER_WITH_COMPARABLE_COLUMN)

# <r,<g,r>>
generic_name_mapper.map_name_with_signature("same_as", ROW_FILTER_WITH_GENERIC_COLUMN)

# <r,<f,<n,r>>>
number_column_name_mapper.map_name_with_signature("filter_number_greater", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
number_column_name_mapper.map_name_with_signature("filter_number_greater_equals",
                                                  ROW_FILTER_WITH_COLUMN_AND_NUMBER)
number_column_name_mapper.map_name_with_signature("filter_number_lesser", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
number_column_name_mapper.map_name_with_signature("filter_number_lesser_equals", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
number_column_name_mapper.map_name_with_signature("filter_number_equals", ROW_FILTER_WITH_COLUMN_AND_NUMBER)
number_column_name_mapper.map_name_with_signature("filter_number_not_equals", ROW_FILTER_WITH_COLUMN_AND_NUMBER)

# <r,<y,<d,r>>>
date_column_name_mapper.map_name_with_signature("filter_date_greater", ROW_FILTER_WITH_COLUMN_AND_DATE)
date_column_name_mapper.map_name_with_signature("filter_date_greater_equals", ROW_FILTER_WITH_COLUMN_AND_DATE)
date_column_name_mapper.map_name_with_signature("filter_date_lesser", ROW_FILTER_WITH_COLUMN_AND_DATE)
date_column_name_mapper.map_name_with_signature("filter_date_lesser_equals", ROW_FILTER_WITH_COLUMN_AND_DATE)
date_column_name_mapper.map_name_with_signature("filter_date_equals", ROW_FILTER_WITH_COLUMN_AND_DATE)
date_column_name_mapper.map_name_with_signature("filter_date_not_equals", ROW_FILTER_WITH_COLUMN_AND_DATE)

# <r,<t,<s,r>>>
string_column_name_mapper.map_name_with_signature("filter_in", ROW_FILTER_WITH_COLUMN_AND_STRING)
string_column_name_mapper.map_name_with_signature("filter_not_in", ROW_FILTER_WITH_COLUMN_AND_STRING)

# <r,r>
generic_name_mapper.map_name_with_signature("first", ROW_FILTER)
generic_name_mapper.map_name_with_signature("last", ROW_FILTER)
generic_name_mapper.map_name_with_signature("previous", ROW_FILTER)
generic_name_mapper.map_name_with_signature("next", ROW_FILTER)

# <r,n>
generic_name_mapper.map_name_with_signature("count", COUNT_TYPE)

# <r,<f,n>>
number_column_name_mapper.map_name_with_signature("max", ROW_NUM_OP)
number_column_name_mapper.map_name_with_signature("min", ROW_NUM_OP)
number_column_name_mapper.map_name_with_signature("average", ROW_NUM_OP)
number_column_name_mapper.map_name_with_signature("sum", ROW_NUM_OP)

# <r,<r,<f,n>>>
number_column_name_mapper.map_name_with_signature("diff", NUM_DIFF_WITH_COLUMN)

# <n,<n,<n,d>>>
generic_name_mapper.map_name_with_signature("date", DATE_FUNCTION_TYPE)

COMMON_NAME_MAPPING = generic_name_mapper.common_name_mapping
COMMON_TYPE_SIGNATURE = generic_name_mapper.common_type_signature
STRING_COLUMN_NAME_MAPPING = string_column_name_mapper.common_name_mapping
STRING_COLUMN_TYPE_SIGNATURE = string_column_name_mapper.common_type_signature
NUMBER_COLUMN_NAME_MAPPING = number_column_name_mapper.common_name_mapping
NUMBER_COLUMN_TYPE_SIGNATURE = number_column_name_mapper.common_type_signature
DATE_COLUMN_NAME_MAPPING = date_column_name_mapper.common_name_mapping
DATE_COLUMN_TYPE_SIGNATURE = date_column_name_mapper.common_type_signature
COMPARABLE_COLUMN_NAME_MAPPING = comparable_column_name_mapper.common_name_mapping
COMPARABLE_COLUMN_TYPE_SIGNATURE = comparable_column_name_mapper.common_type_signature
