"""
Defines all the types in the WikitablesQuestions domain. We exploit the type logic in ``nltk.sem.logic``
here. This module also contains two helper classes that add some functionality on top of NLTK's logic module.
"""
import re

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


class ReverseType(ComplexType):
    """
    ReverseType is a special kind of ``ComplexType`` where type resolution involves matching the return
    type with the reverse of the argument type. So all we care about are the types of the surrounding
    expressions, and return a resolution that matches whatever parts are present in the type signatures
    of the arguments and the return expressions.

    Following are the resolutions for some example type signatures being matched against:
        <?, <e,r>>      :   <<r,e>, <e,r>>
        <<r,?>, <?,e>>  :   <<r,e>, <e,r>>
        <<r,?>, ?>      :   <<r,?>, <?,r>>>
        <<r,?>, <e,?>>  :   None  (causes resolution failure)
    """
    @overrides
    def __eq__(self, other):
        return isinstance(other, ReverseType)

    @overrides
    def matches(self, other):
        return self == other or self == ANY_TYPE or other == ANY_TYPE

    @overrides
    def resolve(self, other):
        if not isinstance(other, ComplexType) and not isinstance(other, ReverseType):
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
    def __str__(self):
        if self == ANY_TYPE:
            return "%s" % ANY_TYPE
        else:
            return "<<#1,#2>,<#2,#1>>"

    @overrides
    def str(self):
        if self == ANY_TYPE:
            return ANY_TYPE.str()
        else:
            return "<<#1,#2> -> <#2,#1>>"


class ReverseApplicationExpression(ApplicationExpression):
    """
    NLTK's ``ApplicationExpression`` (that represents function applications like P(x)) does not
    handle the case where P is reverse (R), which is a special case because the type of R(x) depends on
    the type of x. We override ``ApplicationExpression`` to redefine the type decleration of the application.
    """
    @property
    def type(self):
        argument_type = self.argument.type
        reversed_argument_type = ComplexType(argument_type.second, argument_type.first)
        return reversed_argument_type


class LogicParserWithReverse(LogicParser):
    """
    Since we defined a new kind of ``ApplicationExpression`` above, the ``LogicParser`` should be able to
    create this new kind of expression when needed. We do that by overriding the ``LogicParser`` as well.
    """
    @overrides
    def make_ApplicationExpression(self, function, argument):
        # This gets called when the tree is being built by ``LogicParser.parse``. So, we do not
        # have access to the type signatures yet. Thus, we need to look at the name of the function
        # to define the appropriate kind of ``ApplicationExpression``.
        # Note: We restrict the name of the reverse function to start with upper case R.
        if self.is_reverse(str(function)):
            return ReverseApplicationExpression(function, argument)
        return super(LogicParserWithReverse, self).make_ApplicationExpression(function, argument)

    @staticmethod
    def is_reverse(variable_name):
        return re.match(r"^R\d*$", variable_name)


CELL_TYPE = EntityType()
ROW_TYPE = NamedBasicType("ROW")
# TODO (pradeep): Merging dates and nums. Can define a hierarchy instead.
DATE_NUM_TYPE = NamedBasicType("DATENUM")
COLUMN_TYPE = ComplexType(CELL_TYPE, ROW_TYPE)
DATE_FUNCTION_TYPE = ComplexType(DATE_NUM_TYPE, CELL_TYPE)
MAX_MIN_TYPE = ComplexType(DATE_NUM_TYPE, DATE_NUM_TYPE)
NEXT_ROW_TYPE = ComplexType(ROW_TYPE, ROW_TYPE)
REVERSE_TYPE = ReverseType(ComplexType(ANY_TYPE, ANY_TYPE), ComplexType(ANY_TYPE, ANY_TYPE))
