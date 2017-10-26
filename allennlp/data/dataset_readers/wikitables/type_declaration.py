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
    ``PlaceholderType`` is a ``ComplexType`` that involves placeholders, and thus its type resolution is
    context sensitive. This is an abstract class for all placeholder types like reverse, and, or, argmax, etc.
    The subclasses need to do two things:
        1) Override the property ``_signature`` to define the type signature (this is just the signature's
        string representation and will not affect type inference or checking). You will see this signature in
        action sequences.
        2) Override ``resolve`` to resolve the type appropriately (see the docstring in ``resolve`` for more
        information).
    """
    @property
    def _signature(self):
        raise NotImplementedError

    @overrides
    def resolve(self, other):
        """
        This method is central to type inference and checking. When a variable's type is being checked, we
        compare what we know of its type against what is expected of its type by its context. The expectation
        is provided as ``other``. We make sure that there are no contradictions between this type and other,
        and return an updated type which may be more specific than the original type.

        For example, say this type is of the function variable F in F(cell), and we start out with <?, d> (that
        is, it takes any type and returns d). Now we have already resolved `cell` to be of type `e`. Then
        ``resolve`` gets called with other = <e, ?>, because we know F is a function that took a constant of
        type `e`. When we resolve <e, ?> against <?, d>, there will not be a contradiction, because any type
        can be successfully resolved against ?. Finally we return <e, d> as the resolved type.

        As a counter example, if we are trying to resolve <?, d> against <?, e>, the resolution fails, and in
        that case, this method returns ``None``.

        Note that resolution may be unidirectional because of ?, and so in the subclasses of this type, we
        explicitly resolve in both directions.
        """
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
    @property
    def _signature(self):
        return "<<#1,#2>,<#2,#1>>"

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
    @property
    def _signature(self):
        return "<#1,#1>"

    @overrides
    def resolve(self, other):
        """See ``PlaceholderType.resolve``"""
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
    @property
    def _signature(self):
        return "<#1,<#1,#1>>"

    @overrides
    def resolve(self, other):
        """See ``PlaceholderType.resolve``"""
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
    @property
    def _signature(self):
        return "<d,<d,<#1,<<d,#1>,#1>>>>"

    @overrides
    def resolve(self, other):
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
                       "fb:row.row.next": "N",
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
                       "sum": "S0",
                       "avg": "S1",
                       "-": "F",
                       "x": "X",
                      }

COMMON_TYPE_SIGNATURE = {"R": REVERSE_TYPE,
                         "A0": ARG_EXTREME_TYPE,
                         "A1": ARG_EXTREME_TYPE,
                         "M0": UNARY_NUM_OP_TYPE,
                         "M1": UNARY_NUM_OP_TYPE,
                         ">": UNARY_NUM_OP_TYPE,
                         "<": UNARY_NUM_OP_TYPE,
                         "G": UNARY_NUM_OP_TYPE,
                         "L": UNARY_NUM_OP_TYPE,
                         "S0": UNARY_NUM_OP_TYPE,
                         "S1": UNARY_NUM_OP_TYPE,
                         "F": BINARY_NUM_OP_TYPE,
                         "D": DATE_FUNCTION_TYPE,
                         "B": DATE_FUNCTION_TYPE,
                         "B2": DATE_FUNCTION_TYPE,
                         "I": NUMBER_TYPE,
                         "N": NEXT_ROW_TYPE,
                         "Q": IDENTITY_TYPE,
                         "T": IDENTITY_TYPE,
                         "V": IDENTITY_TYPE,
                         "O": CONJUNCTION_TYPE,
                         "A": CONJUNCTION_TYPE,
                         "W": ROW_INDEX_TYPE,
                         "C": COUNT_TYPE,
                         "T0": ROW_TYPE,
                         "X": ANY_TYPE,
                        }


class DynamicTypeApplicationExpression(ApplicationExpression):
    """
    NLTK's ``ApplicationExpression`` (which represents function applications like P(x)) has two limitations,
    which we overcome by inheriting from ``ApplicationExpression`` and overriding two methods.

    Firstly, ``ApplicationExpression`` does not handle the case where P's type involves placeholders
    (R, V, !=, etc.), which are special cases because their return types depend on the type of their
    arguments (x). We override the property ``type`` to redefine the type of the application.

    Secondly, NLTK's variables only bind to entities, and thus the variable types are 'e' by default. We
    get around this issue by replacing x with a function V(X), whose initial type is ANY_TYPE, and later
    gets resolved based on the type signature of the function whose scope the variable appears in. This
    variable binding operation is implemented by overriding ``_set_type`` below.
    """
    @property
    def type(self):
        # This gets called when the tree is being built by ``LogicParser.parse``. So, we do not
        # have access to the type signatures yet. Thus, we need to look at the name of the function
        # to return the type.
        if not self._has_placeholder(str(self.function)):
            return super(DynamicTypeApplicationExpression, self).type
        if self.function.type == ANY_TYPE:
            return ANY_TYPE
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

    def _set_type(self, other_type=ANY_TYPE, signature=None):
        """
        We override this method to do just one thing on top of ``ApplicationExpression._set_type``. In
        lambda expressions of the form /x F(x), where the function is F and the argument is x, we can use
        the type of F to infer the type of x. That is, if F is of type <a, b>, we can resolve the type of
        x against a. We do this as the additional step after setting the type of F(x).

        So why does NLTK not already do this? NLTK assumes all variables (x) are of type entity (e). So it
        does not have to resolve the type of x anymore. However, this would cause type inference failures in
        our case since x can bind to rows, numbers or cells, each of which has a different type. To deal with
        this issue, we replaced x with V(X) ((var x) in Sempre) and made X of type ANY_TYPE, and V of type
        <#1, #1>. We cannot leave X as ANY_TYPE because that would propagate up the tree. We need to set its
        type when we have the information about F. Hence this method.
        """
        super(DynamicTypeApplicationExpression, self)._set_type(other_type, signature)
        if isinstance(self.argument, ApplicationExpression) and str(self.argument.function) == "V":
            # pylint: disable=protected-access
            self.argument.argument._set_type(self.function.type.first)

    @staticmethod
    def _has_placeholder(variable_name):
        if variable_name not in COMMON_TYPE_SIGNATURE:
            return False
        return COMMON_TYPE_SIGNATURE[variable_name] in [REVERSE_TYPE, IDENTITY_TYPE,
                                                        CONJUNCTION_TYPE, ARG_EXTREME_TYPE]

class DynamicTypeLogicParser(LogicParser):
    """
    Since we defined a new kind of ``ApplicationExpression`` above, the ``LogicParser`` should be able to
    create this new kind of expression. We do that by overriding the ``LogicParser`` as well.
    """
    @overrides
    def make_ApplicationExpression(self, function, argument):
        return DynamicTypeApplicationExpression(function, argument)
