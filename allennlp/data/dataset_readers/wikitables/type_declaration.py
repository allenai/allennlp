"""
Defines all the types in the WikitablesQuestions domain. We exploit the type logic in ``nltk.sem.logic``
here. This module also contains two helper classes that add some functionality on top of NLTK's logic module.
"""
from overrides import overrides

from nltk.sem.logic import ApplicationExpression, BasicType, ComplexType, EntityType, ANY_TYPE, LogicParser


class NamedBasicType(BasicType):
    """
    A ``BasicType`` that also takes the name of the type as an argument to its constructor.

    Parameters
    ----------
    string_rep : str
        String representation of the type.
    """
    def __init__(self, string_rep) -> None:
        self._string_rep = string_rep

    def __str__(self) -> str:
        return self._string_rep.lower()[0]

    def str(self) -> str:
        return self._string_rep

class PlaceholderType(ComplexType):
    """
    This is a ``ComplexType`` that involve placeholders, and thus the type resolution is context sensitive.
    """
    def __init__(self, first, second) -> None:
        super(PlaceholderType, self).__init__(first, second)
        self._signature = "?"

    @overrides
    def resolve(self, other):
        # TODO (pradeep): Generally explain how resolution works.
        raise NotImplementedError

    @overrides
    def __eq__(self, other):
        return self.__class__ == other.__class__

    @overrides
    def matches(self, other):
        return self == other or self == ANY_TYPE or other == ANY_TYPE

    @overrides
    def __str__(self):
        if self == ANY_TYPE:
            return "%s" % ANY_TYPE
        else:
            return self._signature

    @overrides
    def str(self):
        if self == ANY_TYPE:
            return ANY_TYPE.str()
        else:
            return self._signature



class ReverseType(PlaceholderType):
    """
    ReverseType is a kind of ``PlaceholderType`` where type resolution involves matching the return
    type with the reverse of the argument type. So all we care about are the types of the surrounding
    expressions, and return a resolution that matches whatever parts are present in the type signatures
    of the arguments and the return expressions.

    Following are the resolutions for some example type signatures being matched against:
        <?, <e,r>>      :   <<r,e>, <e,r>>
        <<r,?>, <?,e>>  :   <<r,e>, <e,r>>
        <<r,?>, ?>      :   <<r,?>, <?,r>>>
        <<r,?>, <e,?>>  :   None  (causes resolution failure)
    """
    def __init__(self, first, second):
        super(ReverseType, self).__init__(first, second)
        self._signature = "<<#1,#2>,<#2,#1>>"

    @overrides
    def resolve(self, other):
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


class IdentityType(PlaceholderType):
    """
    ``IdentityType`` is a special kind of ``ComplexType`` that takes an argument of any type and returns
    an expression of the same type. That is, type signature is <#1, #1>.
    """
    def __init__(self, first, second):
        super(IdentityType, self).__init__(first, second)
        self._signature = "<#1,#1>"

    @overrides
    def resolve(self, other):
        if not isinstance(other, ComplexType):
            return None
        other_first = other.first.resolve(other.second)
        if not other_first:
            return None
        other_second = other.second.resolve(other_first)
        if not other_second:
            return None
        return IdentityType(other_first, other_second)


class ConjunctionType(PlaceholderType):
    """
    ``ConjunctionType`` takes an entity of any type and returns a function that takes and returns the same
    type. That is, its signature is <#1, <#1, #1>>
    """
    def __init__(self, first, second):
        super(ConjunctionType, self).__init__(first, second)
        self._signature = "<#1,<#1,#1>>"

    @overrides
    def resolve(self, other):
        if not isinstance(other, ComplexType):
            return None
        if not isinstance(other.second, ComplexType):
            return None
        other_first = other.first.resolve(other.second.first)
        other_first = other_first.resolve(other.second.second)
        if not other_first:
            return None
        other_second = other.second.resolve(ComplexType(other_first, other_first))
        if not other_second:
            return None
        return ConjunctionType(other_first, other_second)


class ArgExtremeType(PlaceholderType):
    """
    This is the type for argmax and argmin in Sempre. The type signature is <d,<d,<#1,<<d,#1>,#1>>>>.
    Example: (argmax (number 1) (number 1) (fb:row.row.league fb:cell.usl_a_league) fb:row.row.index),
        meaning, of the subset of rows where league == usl_a_league, find the row with the maximum index.
    """
    def __init__(self, first, second):
        super(ArgExtremeType, self).__init__(first, second)
        self._signature = "<d,<d,<#1,<<d,#1>,#1>>>>"

    @overrides
    def resolve(self, other):
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

            # All three types above should resolve against each other.
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

CELL_TYPE = EntityType()
ROW_TYPE = NamedBasicType("ROW")
# TODO (pradeep): Merging dates and nums. Can define a hierarchy instead.
DATE_NUM_TYPE = NamedBasicType("DATENUM")
# Functions like fb:row.row.year.
COLUMN_TYPE = ComplexType(CELL_TYPE, ROW_TYPE)
# fb:cell.cell.date
DATE_FUNCTION_TYPE = ComplexType(DATE_NUM_TYPE, CELL_TYPE)
# number
NUMBER_TYPE = ComplexType(EntityType(), DATE_NUM_TYPE)
# Numerical operations: max, min, >, <. argmax, argmin etc.
NUM_OP_TYPE = ComplexType(DATE_NUM_TYPE, DATE_NUM_TYPE)
# next
NEXT_ROW_TYPE = ComplexType(ROW_TYPE, ROW_TYPE)
# reverse
REVERSE_TYPE = ReverseType(ComplexType(ANY_TYPE, ANY_TYPE), ComplexType(ANY_TYPE, ANY_TYPE))
# !=, fb:type.object.type
# fb:type.object.type takes a type and returns all objects of that type.
IDENTITY_TYPE = IdentityType(ANY_TYPE, ANY_TYPE)
# var. var (x) can be of any type.
#VARIABLE_TYPE = ComplexType(EntityType(), ANY_TYPE)
# index
ROW_INDEX_TYPE = ComplexType(DATE_NUM_TYPE, ROW_TYPE)
# count
COUNT_TYPE = ComplexType(ANY_TYPE, DATE_NUM_TYPE)
# and, or
CONJUNCTION_TYPE = ConjunctionType(ANY_TYPE, ANY_TYPE)
# argmax, argmin
ARG_EXTREME_TYPE = ArgExtremeType(ANY_TYPE, ANY_TYPE)

COMMON_NAME_MAPPING = {"reverse": "R",
                       "max": "M0",
                       "min": "M1",
                       "argmax": "A0",
                       "argmin": "A1",
                       "and": "A",
                       "or": "O",
                       "next": "N",
                       "number": "I",
                       "lambda": "\\",
                       "var": "V",
                       "fb:cell.cell.date": "D",
                       "fb:cell.cell.number": "B",
                       "fb:cell.cell.num2": "B2",
                       "fb:row.row.index": "W",
                       "fb:type.row": "T0",
                       "fb:type.object.type": "T",
                       "count": "C",
                       "!=": "Q",
                       ">=": "G",
                       "<=": "L",
                      }

COMMON_TYPE_SIGNATURE = {"R": REVERSE_TYPE,
                         "M0": NUM_OP_TYPE,
                         "M1": NUM_OP_TYPE,
                         "A0": ARG_EXTREME_TYPE,
                         "A1": ARG_EXTREME_TYPE,
                         ">": NUM_OP_TYPE,
                         "<": NUM_OP_TYPE,
                         "G": NUM_OP_TYPE,
                         "L": NUM_OP_TYPE,
                         "D": DATE_FUNCTION_TYPE,
                         "B": DATE_FUNCTION_TYPE,
                         "B2": DATE_FUNCTION_TYPE,
                         "I": NUMBER_TYPE,
                         "N": NEXT_ROW_TYPE,
                         "Q": IDENTITY_TYPE,
                         "T": IDENTITY_TYPE,
                         #"V": VARIABLE_TYPE,
                         "V": IDENTITY_TYPE,
                         "O": CONJUNCTION_TYPE,
                         "A": CONJUNCTION_TYPE,
                         "W": ROW_INDEX_TYPE,
                         "C": COUNT_TYPE,
                         "T0": ROW_TYPE,
                        }


class PlaceholderApplicationExpression(ApplicationExpression):
    """
    NLTK's ``ApplicationExpression`` (which represents function applications like P(x)) does not
    handle the case where P's type involves placeholders (R, V, !=, etc.), which are special cases because
    their return types depend on the type of their arguments (x). We override ``ApplicationExpression`` to
    redefine the type of the application.
    """
    @property
    def type(self):
        assert isinstance(self.function.type, (ReverseType, IdentityType, ConjunctionType, ArgExtremeType))
        argument_type = self.argument.type
        if isinstance(self.function.type, ReverseType):
            return_type = ComplexType(argument_type.second, argument_type.first)
        elif isinstance(self.function.type, ConjunctionType):
            return_type = ComplexType(argument_type, argument_type)
        elif isinstance(self.function.type, ArgExtremeType):
            # Returning the first #1 from <d,<d,<#1,<<d,#1>,#1>>>>.
            # This is called after the placeholders are resolved.
            return_type = self.function.type.second.second.first
        else:
            return_type = argument_type
        return return_type


class LogicParserWithPlaceholders(LogicParser):
    """
    Since we defined a new kinds of ``ApplicationExpression``s above, the ``LogicParser`` should be able to
    create these new kinds of expressions when needed. We do that by overriding the ``LogicParser`` as well.
    """
    @overrides
    def make_ApplicationExpression(self, function, argument):
        # This gets called when the tree is being built by ``LogicParser.parse``. So, we do not
        # have access to the type signatures yet. Thus, we need to look at the name of the function
        # to define the appropriate kind of ``ApplicationExpression``.
        if self.has_placeholder(str(function)):
            return PlaceholderApplicationExpression(function, argument)
        return super(LogicParserWithPlaceholders, self).make_ApplicationExpression(function, argument)

    @staticmethod
    def has_placeholder(variable_name):
        if variable_name not in COMMON_TYPE_SIGNATURE:
            return False
        return COMMON_TYPE_SIGNATURE[variable_name] in [REVERSE_TYPE, IDENTITY_TYPE,
                                                        CONJUNCTION_TYPE, ARG_EXTREME_TYPE]
