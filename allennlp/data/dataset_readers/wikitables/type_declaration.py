"""
Defines all the types in the WikitablesQuestions domain. We exploit the type logic in ``nltk.sem.logic``
here. This module also contains two helper classes that add some functionality on top of NLTK's logic module.
"""
from overrides import overrides

from nltk.sem.logic import ApplicationExpression, ConstantExpression, Variable, LogicParser
from nltk.sem.logic import BasicType, ComplexType, EntityType, ANY_TYPE


class NamedBasicType(BasicType):
    """
    A ``BasicType`` that also takes the name of the type as an argument to its constructor. Type resolution
    uses the output of ``__str__`` as well, so basic types with different representations do not resolve
    against each other.

    Parameters
    ----------
    string_rep : str
        String representation of the type.
    """
    def __init__(self, string_rep) -> None:
        self._string_rep = string_rep

    def __str__(self) -> str:
        return self._string_rep.lower()[0]

    def str(self):
        return self._string_rep


class PlaceholderType(ComplexType):
    """
    ``PlaceholderType`` is a ``ComplexType`` that involves placeholders, and thus its type resolution is
    context sensitive. This is an abstract class for all placeholder types like reverse, and, or, argmax, etc.

    Note that ANY_TYPE in NLTK's type system doesn't work like a wild card. Once the type of a variable gets
    resolved to a specific type, NLTK changes the type of that variable to that specific type. Hence, what
    NLTK calls "ANY_TYPE", is essentially a "yet-to-be-decided type". This is a problem because we may want the
    same variable to bind to different types within a logical form, and using ANY_TYPE for this purpose will
    cause a resolution failure. For example the count function may apply to both rows and cells in the same
    logical form, and making count of type ``ComplexType(ANY_TYPE, DATE_NUM_TYPE)`` will cause a resolution
    error. This class lets you define ``ComplexType``s with placeholders that are actually wild cards.

    The subclasses of this abstract class need to do two things:
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

        Note that a successful resolution does not imply equality of types because of one of them may be
        ANY_TYPE, and so in the subclasses of this type, we explicitly resolve in both directions.
        """
        raise NotImplementedError

    @overrides
    def __eq__(self, other):
        return self.__class__ == other.__class__

    @overrides
    def matches(self, other):
        # self == ANY_TYPE = True iff self.first == ANY_TYPE and self.second == ANY_TYPE.
        return self == other or self == ANY_TYPE or other == ANY_TYPE

    @overrides
    def __str__(self):
        if self == ANY_TYPE:
            # If the type remains unresolved, we return `?` instead of its signature.
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
        <<r,?>, <e,?>>  :   <<r,e>, <e,r>>
        <<r,?>, ?>      :   <<r,?>, <?,r>>>
        <<r,?>, <?,e>>  :   None  (causes resolution failure)
    """
    @property
    def _signature(self):
        return "<<#1,#2>,<#2,#1>>"

    @overrides
    def resolve(self, other):
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


class IdentityType(PlaceholderType):
    """
    ``IdentityType`` is a kind of ``PlaceholderType`` that takes an argument of any type and returns
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
        if other_first is None:
            return None
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


class CountType(PlaceholderType):
    """
    Type of a function that counts arbitrary things. Signature is <#1,d>.
    """
    @property
    def _signature(self):
        return "<#1,d>"

    @overrides
    def resolve(self, other):
        """See ``PlaceholderType.resolve``"""
        if not isinstance(other, ComplexType):
            return None
        resolved_second = DATE_NUM_TYPE.resolve(other.second)
        if not resolved_second:
            return None
        return CountType(ANY_TYPE, resolved_second)


CELL_TYPE = EntityType()
PART_TYPE = NamedBasicType("PART")
ROW_TYPE = NamedBasicType("ROW")
# TODO (pradeep): Merging dates and nums. Can define a hierarchy instead.
DATE_NUM_TYPE = NamedBasicType("DATENUM")
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
ARG_EXTREME_TYPE = ArgExtremeType(ANY_TYPE, ANY_TYPE)

COMMON_NAME_MAPPING = {"lambda": "\\"}

COMMON_TYPE_SIGNATURE = {}

def _add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature

_add_common_name_with_type("reverse", "R", REVERSE_TYPE)
_add_common_name_with_type("argmax", "A0", ARG_EXTREME_TYPE)
_add_common_name_with_type("argmin", "A1", ARG_EXTREME_TYPE)
_add_common_name_with_type("max", "M0", UNARY_NUM_OP_TYPE)
_add_common_name_with_type("min", "M1", UNARY_NUM_OP_TYPE)
_add_common_name_with_type("and", "A", CONJUNCTION_TYPE)
_add_common_name_with_type("or", "O", CONJUNCTION_TYPE)
_add_common_name_with_type("fb:row.row.next", "N", NEXT_ROW_TYPE)
_add_common_name_with_type("number", "I", NUMBER_FUNCTION_TYPE)
_add_common_name_with_type("date", "D0", DATE_FUNCTION_TYPE)
_add_common_name_with_type("var", "V", IDENTITY_TYPE)
_add_common_name_with_type("fb:cell.cell.part", "P", PART2CELL_TYPE)
_add_common_name_with_type("fb:cell.cell.date", "D1", CELL2DATE_NUM_TYPE)
_add_common_name_with_type("fb:cell.cell.number", "I1", CELL2DATE_NUM_TYPE)
_add_common_name_with_type("fb:cell.cell.num2", "I2", CELL2DATE_NUM_TYPE)
_add_common_name_with_type("fb:row.row.index", "W", ROW_INDEX_TYPE)
_add_common_name_with_type("fb:type.row", "T0", ROW_TYPE)
_add_common_name_with_type("fb:type.object.type", "T", IDENTITY_TYPE)
_add_common_name_with_type("count", "C", COUNT_TYPE)
_add_common_name_with_type("!=", "Q", IDENTITY_TYPE)
_add_common_name_with_type(">", "G0", UNARY_NUM_OP_TYPE)
_add_common_name_with_type(">=", "G1", UNARY_NUM_OP_TYPE)
_add_common_name_with_type("<", "L0", UNARY_NUM_OP_TYPE)
_add_common_name_with_type("<=", "L1", UNARY_NUM_OP_TYPE)
_add_common_name_with_type("sum", "S0", UNARY_NUM_OP_TYPE)
_add_common_name_with_type("avg", "S1", UNARY_NUM_OP_TYPE)
_add_common_name_with_type("-", "F", BINARY_NUM_OP_TYPE)  # subtraction
_add_common_name_with_type("x", "X", ANY_TYPE)


class TypedConstantExpression(ConstantExpression):
    # pylint: disable=abstract-method
    """
    NLTK assumes all constants are of type ``EntityType`` (e) by default. We define this new class where we
    can pass a default type to the constructor and use that in the ``_set_type`` method.
    """
    def __init__(self, variable, default_type):
        super(TypedConstantExpression, self).__init__(variable)
        self._default_type = default_type

    @overrides
    def _set_type(self, other_type=ANY_TYPE, signature=None):
        if other_type == ANY_TYPE:
            super(TypedConstantExpression, self)._set_type(self._default_type, signature)
        else:
            super(TypedConstantExpression, self)._set_type(other_type, signature)


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
            # Returning <d,<#1,<<d,#1>,#1>>>.
            # This is called after the placeholders are resolved.
            return_type = self.function.type.second
        else:
            # The function is of type IdentityType.
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

    @overrides
    def make_VariableExpression(self, name):
        if name.startswith("part:"):
            return TypedConstantExpression(Variable(name), PART_TYPE)
        return super(DynamicTypeLogicParser, self).make_VariableExpression(name)
