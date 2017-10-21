"""
Defines all the types in the WikitablesQuestions domain. We exploit the type logic in ``nltk.sem.logic``
here. This module also contains two helper classes that add some functionality on top of NLTK's logic module.
"""
from overrides import overrides

from nltk.sem.logic import BasicType, ComplexType, EntityType, ANY_TYPE


class BasicTypeFactory(BasicType):
    """
    A factory class that takes a string representation of a type and returns a new ``BasicType``.

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

    @classmethod
    def define_type(cls, string_rep: str) -> 'BasicTypeFactory':
        return cls(string_rep)


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
    def resolve(self, other):
        if not isinstance(other, ComplexType):
            return None

        # other.first and other.second are the argument and return types respectively.
        reversed_second = ComplexType(other.second.second, other.second.first)
        first = other.first.resolve(reversed_second)
        reversed_first = ComplexType(first.second, first.first)
        second = other.second.resolve(reversed_first)
        if first and second:
            return ReverseType(first, second)
        return None


class WikitablesTypeDeclaration:
    """
    Defines the types in the WikitablesQuestions domain. This type system uses ``nltk.sem.logic.Type``.
    """
    CELL_TYPE = EntityType()
    ROW_TYPE = BasicTypeFactory.define_type("ROW")
    # TODO: Merging dates and nums. Can define a hierarchy instead.
    DATE_NUM_TYPE = BasicTypeFactory.define_type("DATENUM")
    COLUMN_TYPE = ComplexType(CELL_TYPE, ROW_TYPE)
    DATE_FUNCTION_TYPE = ComplexType(CELL_TYPE, DATE_NUM_TYPE)
    MAX_MIN_TYPE = ComplexType(DATE_NUM_TYPE, DATE_NUM_TYPE)
    NEXT_ROW_TYPE = ComplexType(ROW_TYPE, ROW_TYPE)
    REVERSE_TYPE = ReverseType(ComplexType(ANY_TYPE, ANY_TYPE), ComplexType(ANY_TYPE, ANY_TYPE))
