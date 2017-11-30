"""
This module defines some classes that are generally useful for defining a type system for a new domain. We
inherit the type logic in ``nltk.sem.logic`` and add some functionality on top of it here. There are two main
improvements:
1) Firstly, we allow defining multiple basic types with their own names (see ``NamedBasicType``).
2) Secondly, we allow defining function types that have placeholders in them (see ``PlaceholderType``).
We also extend NLTK's ``LogicParser`` to define a ``DynamicTypeLogicParser`` that knows how to deal with the
two improvements above.
"""
from typing import Dict, Set, Optional
from overrides import overrides

from nltk.sem.logic import Expression, ApplicationExpression, ConstantExpression, LogicParser, Variable
from nltk.sem.logic import Type, BasicType, ComplexType, ANY_TYPE


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

    def __str__(self):
        # TODO (pradeep): This limits the number of basic types we can have to 26. We may want to change this
        # in the future if we extend to domains where we have more than 26 basic types.
        return self._string_rep.lower()[0]

    def str(self):
        return self._string_rep


class PlaceholderType(ComplexType):
    """
    ``PlaceholderType`` is a ``ComplexType`` that involves placeholders, and thus its type resolution is
    context sensitive. This is an abstract class for all placeholder types like reverse, and, or, argmax, etc.

    Note that ANY_TYPE in NLTK's type system doesn't work like a wild card. Once the type of a variable gets
    resolved to a specific type, NLTK changes the type of that variable to that specific type. Hence, what
    NLTK calls "ANY_TYPE", is essentially a "yet-to-be-decided" type. This is a problem because we may want the
    same variable to bind to different types within a logical form, and using ANY_TYPE for this purpose will
    cause a resolution failure. For example the count function may apply to both rows and cells in the same
    logical form, and making count of type ``ComplexType(ANY_TYPE, DATE_NUM_TYPE)`` will cause a resolution
    error. This class lets you define ``ComplexType`` s with placeholders that are actually wild cards.

    The subclasses of this abstract class need to do three things
    1) Override the property ``_signature`` to define the type signature (this is just the signature's
    string representation and will not affect type inference or checking). You will see this signature in
    action sequences.
    2) Override ``resolve`` to resolve the type appropriately (see the docstring in ``resolve`` for more
    information).
    3) Override ``get_application_type`` which returns the return type when this type is applied as a function
    to an argument of a specified type.
    For example, if you defined a reverse type by inheriting from this class, ``get_application_type`` gets an
    argument of type ``<a,b>``, it should return ``<b,a>`` .
    """
    @property
    def _signature(self) -> str:
        raise NotImplementedError

    @overrides
    def resolve(self, other: Type) -> Optional[Type]:
        """
        This method is central to type inference and checking. When a variable's type is being checked, we
        compare what we know of its type against what is expected of its type by its context. The expectation
        is provided as ``other``. We make sure that there are no contradictions between this type and other,
        and return an updated type which may be more specific than the original type.

        For example, say this type is of the function variable F in F(cell), and we start out with ``<?, d>``
        (that is, it takes any type and returns ``d`` ). Now we have already resolved cell to be of type
        ``e`` . Then ``resolve`` gets called with ``other = <e, ?>`` , because we know F is a function that
        took a constant of type ``e`` . When we resolve ``<e, ?>`` against ``<?, d>`` , there will not be a
        contradiction, because any type can be successfully resolved against ``?`` . Finally we return
        ``<e, d>`` as the resolved type.

        As a counter example, if we are trying to resolve ``<?, d>`` against ``<?, e>`` , the resolution fails,
        and in that case, this method returns ``None`` .

        Note that a successful resolution does not imply equality of types because of one of them may be
        ANY_TYPE, and so in the subclasses of this type, we explicitly resolve in both directions.
        """
        raise NotImplementedError

    def get_application_type(self, argument_type: Type) -> Type:
        """
        This method returns the resulting type when this type is applied as a function to an argument of
        the given type.
        """
        raise NotImplementedError

    @overrides
    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__

    @overrides
    def matches(self, other) -> bool:
        # self == ANY_TYPE = True iff self.first == ANY_TYPE and self.second == ANY_TYPE.
        return self == other or self == ANY_TYPE or other == ANY_TYPE

    @overrides
    def __str__(self):
        if self == ANY_TYPE:
            # If the type remains unresolved, we return ? instead of its signature.
            return "%s" % ANY_TYPE
        else:
            return self._signature

    @overrides
    def str(self):
        if self == ANY_TYPE:
            return ANY_TYPE.str()
        else:
            return self._signature


class IdentityType(PlaceholderType):
    """
    ``IdentityType`` is a kind of ``PlaceholderType`` that takes an argument of any type and returns
    an expression of the same type. That is, type signature is <#1, #1>. This is in this module because it is
    a commonly needed ``PlaceholderType`` in many domains. For example, if your logical form language has
    lambda expressions, it is quite convenient to specify the variable's usage as "(var x)", and you can make
    "var" a function of this type.
    """
    @property
    def _signature(self) -> str:
        return "<#1,#1>"

    @overrides
    def resolve(self, other) -> Optional[Type]:
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

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        return argument_type


class TypedConstantExpression(ConstantExpression):
    # pylint: disable=abstract-method
    """
    NLTK assumes all constants are of type ``EntityType`` (e) by default. We define this new class where we
    can pass a default type to the constructor and use that in the ``_set_type`` method.
    """
    def __init__(self, variable, default_type: Type) -> None:
        super(TypedConstantExpression, self).__init__(variable)
        self._default_type = default_type

    @overrides
    def _set_type(self, other_type=ANY_TYPE, signature=None) -> None:
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
    def __init__(self, function: Expression, argument: Expression, variables_with_placeholders: Set[str]) -> None:
        super(DynamicTypeApplicationExpression, self).__init__(function, argument)
        self._variables_with_placeholders = variables_with_placeholders

    @property
    def type(self):
        # This gets called when the tree is being built by ``LogicParser.parse``. So, we do not
        # have access to the type signatures yet. Thus, we need to look at the name of the function
        # to return the type.
        if not str(self.function) in self._variables_with_placeholders:
            return super(DynamicTypeApplicationExpression, self).type
        if self.function.type == ANY_TYPE:
            return ANY_TYPE
        argument_type = self.argument.type
        return self.function.type.get_application_type(argument_type)

    def _set_type(self, other_type: Type = ANY_TYPE, signature=None) -> None:
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
        # TODO(pradeep): Assuming the mapping of "var" function is "V". Do something better.
        if isinstance(self.argument, ApplicationExpression) and str(self.argument.function) == "V":
            # pylint: disable=protected-access
            self.argument.argument._set_type(self.function.type.first)


class DynamicTypeLogicParser(LogicParser):
    """
    ``DynamicTypeLogicParser`` is a ``LogicParser`` that can deal with ``NamedBasicType`` and
    ``PlaceholderType`` appropriately. Our extension here does two things differently.

    Firstly, we should handle constants of different types. We do this by passing a dict of format
    ``{name_prefix: type}`` to the constructor. For example, your domain has entities of types unicorns
    and elves, and you have an entity "Phil" of type unicorn, and "Bob" of type "elf". The names of the two
    entities should then be "unicorn:phil" and "elf:bob" respectively.

    Secondly, since we defined a new kind of ``ApplicationExpression`` above, the ``LogicParser`` should be
    able to create this new kind of expression.
    """
    def __init__(self,
                 type_check: bool = True,
                 constant_type_prefixes: Dict[str, BasicType] = None,
                 type_signatures: Dict[str, Type] = None) -> None:
        super(DynamicTypeLogicParser, self).__init__(type_check)
        self._constant_type_prefixes = constant_type_prefixes or {}
        self._variables_with_placeholders = set([name for name, _type in type_signatures.items()
                                                 if isinstance(_type, PlaceholderType)])

    @overrides
    def make_ApplicationExpression(self, function, argument):
        return DynamicTypeApplicationExpression(function, argument, self._variables_with_placeholders)

    @overrides
    def make_VariableExpression(self, name):
        if ":" in name:
            prefix = name.split(":")[0]
            if prefix in self._constant_type_prefixes:
                return TypedConstantExpression(Variable(name), self._constant_type_prefixes[prefix])
            else:
                raise RuntimeError("Unknown prefix: %s. Did you forget to pass it to the constructor?" % prefix)
        return super(DynamicTypeLogicParser, self).make_VariableExpression(name)
