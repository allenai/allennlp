"""
This module defines some classes that are generally useful for defining a type system for a new
domain. We inherit the type logic in ``nltk.sem.logic`` and add some functionality on top of it
here. There are two main improvements:
1) Firstly, we allow defining multiple basic types with their own names (see ``NamedBasicType``).
2) Secondly, we allow defining function types that have placeholders in them (see
``PlaceholderType``).
We also extend NLTK's ``LogicParser`` to define a ``DynamicTypeLogicParser`` that knows how to deal
with the two improvements above.
"""
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import itertools

from overrides import overrides
from nltk.sem.logic import Expression, ApplicationExpression, ConstantExpression, LogicParser, Variable
from nltk.sem.logic import Type, BasicType, ComplexType as NltkComplexType, ANY_TYPE

from allennlp.common.util import START_SYMBOL


class ComplexType(NltkComplexType):
    """
    In NLTK, a ``ComplexType`` is a function.  These functions are curried, so if you need multiple
    arguments for your function you nest ``ComplexTypes``.  That currying makes things difficult
    for us, and we mitigate the problems by adding ``return_type`` and ``argument_type`` functions
    to ``ComplexType``.
    """
    def return_type(self) -> Type:
        """
        Gives the final return type for this function.  If the function takes a single argument,
        this is just ``self.second``.  If the function takes multiple arguments and returns a basic
        type, this should be the final ``.second`` after following all complex types.  That is the
        implementation here in the base class.  If you have a higher-order function that returns a
        function itself, you need to override this method.
        """
        return_type = self.second
        while isinstance(return_type, ComplexType):
            return_type = return_type.second
        return return_type

    def argument_types(self) -> List[Type]:
        """
        Gives the types of all arguments to this function.  For functions returning a basic type,
        we grab all ``.first`` types until ``.second`` is no longer a ``ComplexType``.  That logic
        is implemented here in the base class.  If you have a higher-order function that returns a
        function itself, you need to override this method.
        """
        arguments = [self.first]
        remaining_type = self.second
        while isinstance(remaining_type, ComplexType):
            arguments.append(remaining_type.first)
            remaining_type = remaining_type.second
        return arguments

    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        """
        Takes a set of ``BasicTypes`` and replaces any instances of ``ANY_TYPE`` inside this
        complex type with each of those basic types.
        """
        substitutions = []
        for first_type in substitute_any_type(self.first, basic_types):
            for second_type in substitute_any_type(self.second, basic_types):
                substitutions.append(self.__class__(first_type, second_type))
        return substitutions


class HigherOrderType(ComplexType):
    """
    A higher-order function is a ``ComplexType`` that returns functions.  We just override
    ``return_type`` and ``argument_types`` to make sure that these types are correct.

    Parameters
    ----------
    num_arguments : ``int``
        How many arguments this function takes before returning a function.  We'll go through this
        many levels of nested ``ComplexTypes`` before returning the final ``.second`` as our return
        type.
    first : ``Type``
        Passed to NLTK's ComplexType.
    second : ``Type``
        Passed to NLTK's ComplexType.
    """
    def __init__(self, num_arguments: int, first: Type, second: Type) -> None:
        super().__init__(first, second)
        self.num_arguments = num_arguments

    @overrides
    def return_type(self) -> Type:
        return_type = self.second
        for _ in range(self.num_arguments - 1):
            return_type = return_type.second
        return return_type

    @overrides
    def argument_types(self) -> List[Type]:
        arguments = [self.first]
        remaining_type = self.second
        for _ in range(self.num_arguments - 1):
            arguments.append(remaining_type.first)
            remaining_type = remaining_type.second
        return arguments


class NamedBasicType(BasicType):
    """
    A ``BasicType`` that also takes the name of the type as an argument to its constructor. Type
    resolution uses the output of ``__str__`` as well, so basic types with different
    representations do not resolve against each other.

    Parameters
    ----------
    string_rep : ``str``
        String representation of the type.
    """
    def __init__(self, string_rep) -> None:
        self._string_rep = string_rep

    def __str__(self):
        # TODO (pradeep): This limits the number of basic types we can have to 26. We may want to
        # change this in the future if we extend to domains where we have more than 26 basic types.
        if self._string_rep == START_SYMBOL:
            return START_SYMBOL
        else:
            return self._string_rep.lower()[0]

    def str(self):
        return self._string_rep


class MultiMatchNamedBasicType(NamedBasicType):
    """
    A ``NamedBasicType`` that matches with any type within a list of ``BasicTypes`` that it takes
    as an additional argument during instantiation. We just override the ``matches`` method in
    ``BasicType`` to match against any of the types given by the list.

    Parameters
    ----------
    string_rep : ``str``
        String representation of the type, passed to super class.
    types_to_match : ``List[BasicType]``
        List of types that this type should match with.
    """
    def __init__(self,
                 string_rep,
                 types_to_match: List[BasicType]) -> None:
        super().__init__(string_rep)
        self.types_to_match = set(types_to_match)

    @overrides
    def matches(self, other):
        return super().matches(other) or other in self.types_to_match


class PlaceholderType(ComplexType):
    """
    ``PlaceholderType`` is a ``ComplexType`` that involves placeholders, and thus its type
    resolution is context sensitive. This is an abstract class for all placeholder types like
    reverse, and, or, argmax, etc.

    Note that ANY_TYPE in NLTK's type system doesn't work like a wild card. Once the type of a
    variable gets resolved to a specific type, NLTK changes the type of that variable to that
    specific type. Hence, what NLTK calls "ANY_TYPE", is essentially a "yet-to-be-decided" type.
    This is a problem because we may want the same variable to bind to different types within a
    logical form, and using ANY_TYPE for this purpose will cause a resolution failure. For example
    the count function may apply to both rows and cells in the same logical form, and making count
    of type ``ComplexType(ANY_TYPE, DATE_NUM_TYPE)`` will cause a resolution error. This class lets
    you define ``ComplexType`` s with placeholders that are actually wild cards.

    The subclasses of this abstract class need to do three things
    1) Override the property ``_signature`` to define the type signature (this is just the
    signature's string representation and will not affect type inference or checking). You will see
    this signature in action sequences.
    2) Override ``resolve`` to resolve the type appropriately (see the docstring in ``resolve`` for
    more information).
    3) Override ``get_application_type`` which returns the return type when this type is applied as
    a function to an argument of a specified type.  For example, if you defined a reverse type by
    inheriting from this class, ``get_application_type`` gets an argument of type ``<a,b>``, it
    should return ``<b,a>`` .
    """
    _signature: str = None

    @overrides
    def resolve(self, other: Type) -> Optional[Type]:
        """
        This method is central to type inference and checking. When a variable's type is being
        checked, we compare what we know of its type against what is expected of its type by its
        context. The expectation is provided as ``other``. We make sure that there are no
        contradictions between this type and other, and return an updated type which may be more
        specific than the original type.

        For example, say this type is of the function variable F in F(cell), and we start out with
        ``<?, d>`` (that is, it takes any type and returns ``d`` ). Now we have already resolved
        cell to be of type ``e`` . Then ``resolve`` gets called with ``other = <e, ?>`` , because
        we know F is a function that took a constant of type ``e`` . When we resolve ``<e, ?>``
        against ``<?, d>`` , there will not be a contradiction, because any type can be
        successfully resolved against ``?`` . Finally we return ``<e, d>`` as the resolved type.

        As a counter example, if we are trying to resolve ``<?, d>`` against ``<?, e>`` , the
        resolution fails, and in that case, this method returns ``None`` .

        Note that a successful resolution does not imply equality of types because of one of them
        may be ANY_TYPE, and so in the subclasses of this type, we explicitly resolve in both
        directions.
        """
        raise NotImplementedError

    def get_application_type(self, argument_type: Type) -> Type:
        """
        This method returns the resulting type when this type is applied as a function to an argument of
        the given type.
        """
        raise NotImplementedError

    @overrides
    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        """
        Placeholders mess with substitutions, so even though this method is implemented in the
        superclass, we override it here with a ``NotImplementedError`` to be sure that subclasses
        think about what the right thing to do here is, and do it correctly.
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
            return str(ANY_TYPE)
        else:
            return self._signature

    @overrides
    def str(self):
        if self == ANY_TYPE:
            return ANY_TYPE.str()
        else:
            return self._signature

    __hash__ = ComplexType.__hash__


class UnaryOpType(PlaceholderType):
    """
    ``UnaryOpType`` is a kind of ``PlaceholderType`` that takes an argument of any type and returns
    an expression of the same type.  ``identity`` is an example of this kind of function.  The type
    signature of ``UnaryOpType`` is <#1, #1>.

    Parameters
    ----------
    allowed_substitutions : ``Set[BasicType]``, optional (default=None)
        If given, this sets restrictions on the types that can be substituted.  That is, say you
        have a unary operation that is only permitted for numbers and dates, you can pass those in
        here, and we will only consider those types when calling :func:`substitute_any_type`.  If
        this is ``None``, all basic types are allowed.
    signature : ``str``, optional (default='<#1,#1>')
        The signature of the operation is what will appear in action sequences that include this
        type.  The default value is suitable for functions that apply to any type.  If you have a
        restricted set of allowed substitutions, you likely want to change the type signature to
        reflect that.
    """
    def __init__(self,
                 type_: BasicType = ANY_TYPE,
                 allowed_substitutions: Set[BasicType] = None,
                 signature: str = '<#1,#1>') -> None:
        super().__init__(type_, type_)
        self._allowed_substitutions = allowed_substitutions
        self._signature = signature

    @overrides
    def resolve(self, other) -> Optional[Type]:
        """See ``PlaceholderType.resolve``"""
        if not isinstance(other, NltkComplexType):
            return None
        other_first = other.first.resolve(other.second)
        if not other_first:
            return None
        other_second = other.second.resolve(other_first)
        if not other_second:
            return None
        return UnaryOpType(other_first, self._allowed_substitutions, self._signature)

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        return argument_type

    @overrides
    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        if self.first != ANY_TYPE:
            return [self]
        allowed_basic_types = self._allowed_substitutions if self._allowed_substitutions else basic_types
        return [UnaryOpType(basic_type, self._allowed_substitutions, self._signature)
                for basic_type in allowed_basic_types]


class BinaryOpType(PlaceholderType):
    """
    ``BinaryOpType`` is a function that takes two arguments of the same type and returns an
    argument of that type.  ``+``, ``-``, ``and`` and ``or`` are examples of this kind of function.
    The type signature of ``BinaryOpType`` is ``<#1,<#1,#1>>``.

    Parameters
    ----------
    allowed_substitutions : ``Set[BasicType]``, optional (default=None)
        If given, this sets restrictions on the types that can be substituted.  That is, say you
        have a unary operation that is only permitted for numbers and dates, you can pass those in
        here, and we will only consider those types when calling :func:`substitute_any_type`.  If
        this is ``None``, all basic types are allowed.
    signature : ``str``, optional (default='<#1,<#1,#1>>')
        The signature of the operation is what will appear in action sequences that include this
        type.  The default value is suitable for functions that apply to any type.  If you have a
        restricted set of allowed substitutions, you likely want to change the type signature to
        reflect that.
    """
    def __init__(self,
                 type_: BasicType = ANY_TYPE,
                 allowed_substitutions: Set[BasicType] = None,
                 signature: str = '<#1,<#1,#1>>') -> None:
        super().__init__(type_, ComplexType(type_, type_))
        self._allowed_substitutions = allowed_substitutions
        self._signature = signature

    @overrides
    def resolve(self, other: Type) -> Optional[Type]:
        """See ``PlaceholderType.resolve``"""
        if not isinstance(other, NltkComplexType):
            return None
        if not isinstance(other.second, NltkComplexType):
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
        return BinaryOpType(other_first, self._allowed_substitutions, self._signature)

    @overrides
    def get_application_type(self, argument_type: Type) -> Type:
        return ComplexType(argument_type, argument_type)

    @overrides
    def substitute_any_type(self, basic_types: Set[BasicType]) -> List[Type]:
        if self.first != ANY_TYPE:
            return [self]
        allowed_basic_types = self._allowed_substitutions if self._allowed_substitutions else basic_types
        return [BinaryOpType(basic_type, self._allowed_substitutions, self._signature)
                for basic_type in allowed_basic_types]


class TypedConstantExpression(ConstantExpression):
    # pylint: disable=abstract-method
    """
    NLTK assumes all constants are of type ``EntityType`` (e) by default. We define this new class
    where we can pass a default type to the constructor and use that in the ``_set_type`` method.
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
    NLTK's ``ApplicationExpression`` (which represents function applications like P(x)) has two
    limitations, which we overcome by inheriting from ``ApplicationExpression`` and overriding two
    methods.

    Firstly, ``ApplicationExpression`` does not handle the case where P's type involves
    placeholders (R, V, !=, etc.), which are special cases because their return types depend on the
    type of their arguments (x). We override the property ``type`` to redefine the type of the
    application.

    Secondly, NLTK's variables only bind to entities, and thus the variable types are 'e' by
    default. We get around this issue by replacing x with X, whose initial type is ANY_TYPE, and
    later gets resolved based on the type signature of the function whose scope the variable
    appears in. This variable binding operation is implemented by overriding ``_set_type`` below.
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
        We override this method to do just one thing on top of ``ApplicationExpression._set_type``.
        In lambda expressions of the form /x F(x), where the function is F and the argument is x,
        we can use the type of F to infer the type of x. That is, if F is of type <a, b>, we can
        resolve the type of x against a. We do this as the additional step after setting the type
        of F(x).

        So why does NLTK not already do this? NLTK assumes all variables (x) are of type entity
        (e).  So it does not have to resolve the type of x anymore. However, this would cause type
        inference failures in our case since x can bind to rows, numbers or cells, each of which
        has a different type. To deal with this issue, we made X of type ANY_TYPE. Also, LambdaDCS
        (and some other languages) contain a var function that indicate the usage of variables
        within lambda functions. We map var to V, and made it of type <#1, #1>. We cannot leave X
        as ANY_TYPE because that would propagate up the tree. We need to set its type when we have
        the information about F. Hence this method. Note that the language may or may not contain
        the var function. We deal with both cases below.
        """
        super(DynamicTypeApplicationExpression, self)._set_type(other_type, signature)
        # TODO(pradeep): Assuming the mapping of "var" function is "V". Do something better.
        if isinstance(self.argument, ApplicationExpression) and str(self.argument.function) == "V":
            # pylint: disable=protected-access
            self.argument.argument._set_type(self.function.type.first)
        if str(self.argument) == "X" and str(self.function) != "V":
            # pylint: disable=protected-access
            self.argument._set_type(self.function.type.first)


class DynamicTypeLogicParser(LogicParser):
    """
    ``DynamicTypeLogicParser`` is a ``LogicParser`` that can deal with ``NamedBasicType`` and
    ``PlaceholderType`` appropriately. Our extension here does two things differently.

    Firstly, we should handle constants of different types. We do this by passing a dict of format
    ``{name_prefix: type}`` to the constructor. For example, your domain has entities of types
    unicorns and elves, and you have an entity "Phil" of type unicorn, and "Bob" of type "elf". The
    names of the two entities should then be "unicorn:phil" and "elf:bob" respectively.

    Secondly, since we defined a new kind of ``ApplicationExpression`` above, the ``LogicParser``
    should be able to create this new kind of expression.
    """
    def __init__(self,
                 type_check: bool = True,
                 constant_type_prefixes: Dict[str, BasicType] = None,
                 type_signatures: Dict[str, Type] = None) -> None:
        super(DynamicTypeLogicParser, self).__init__(type_check)
        self._constant_type_prefixes = constant_type_prefixes or {}
        self._variables_with_placeholders = set([name for name, type_ in type_signatures.items()
                                                 if isinstance(type_, PlaceholderType)])

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
                raise RuntimeError(f"Unknown prefix: {prefix}. Did you forget to pass it to the constructor?")
        return super(DynamicTypeLogicParser, self).make_VariableExpression(name)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


class NameMapper:
    """
    The ``LogicParser`` we use has some naming conventions for functions (i.e. they should start
    with an upper case letter, and the remaining characters can only be digits). This means that we
    have to internally represent functions with unintuitive names. This class will automatically
    give unique names following the convention, and populate central mappings with these names. If
    for some reason you need to manually define the alias, you can do so by passing an alias to
    `map_name_with_signature`.

    Parameters
    ----------
    language_has_lambda : ``bool`` (optional, default=False)
        If your language has lambda functions, the word "lambda" needs to be in the name mapping,
        mapped to the alias "\". NLTK understands this symbol, and it doesn't need a type signature
        for it. Setting this flag to True adds the mapping to `name_mapping`.
    alias_prefix : ``str`` (optional, default="F")
        The one letter prefix used for all aliases. You do not need to specify it if you have only
        instance of this class for you language. If not, you can specify a different prefix for each
        name mapping you use for your language.
    """
    def __init__(self,
                 language_has_lambda: bool = False,
                 alias_prefix: str = "F") -> None:
        self.name_mapping: Dict[str, str] = {}
        if language_has_lambda:
            self.name_mapping["lambda"] = "\\"
        self.type_signatures: Dict[str, Type] = {}
        assert len(alias_prefix) == 1 and alias_prefix.isalpha(), (f"Invalid alias prefix: {alias_prefix}"
                                                                   "Needs to be a single upper case character.")
        self._alias_prefix = alias_prefix.upper()
        self._name_counter = 0

    def map_name_with_signature(self,
                                name: str,
                                signature: Type,
                                alias: str = None) -> None:
        if name in self.name_mapping:
            alias = self.name_mapping[name]
            old_signature = self.type_signatures[alias]
            if old_signature != signature:
                raise RuntimeError(f"{name} already added with signature {old_signature}. "
                                   f"Cannot add it again with {signature}!")
        else:
            alias = alias or f"{self._alias_prefix}{self._name_counter}"
            self._name_counter += 1
            self.name_mapping[name] = alias
            self.type_signatures[alias] = signature

    def get_alias(self, name: str) -> str:
        if name not in self.name_mapping:
            raise RuntimeError(f"Unmapped name: {name}")
        return self.name_mapping[name]

    def get_signature(self, name: str) -> Type:
        alias = self.get_alias(name)
        return self.type_signatures[alias]


def substitute_any_type(type_: Type, basic_types: Set[BasicType]) -> List[Type]:
    """
    Takes a type and a set of basic types, and substitutes all instances of ANY_TYPE with all
    possible basic types and returns a list with all possible combinations.  Note that this
    substitution is unconstrained.  That is, If you have a type with placeholders, <#1,#1> for
    example, this may substitute the placeholders with different basic types. In that case, you'd
    want to use ``_substitute_placeholder_type`` instead.
    """
    if type_ == ANY_TYPE:
        return list(basic_types)
    if isinstance(type_, BasicType):
        return [type_]
    # If we've made it this far, we have a ComplexType, and we can just call
    # `type_.substitute_any_type()`.
    return type_.substitute_any_type(basic_types)


def _make_production_string(source: Type, target: Union[List[Type], Type]) -> str:
    return f"{source} -> {target}"


def _get_complex_type_production(complex_type: ComplexType,
                                 multi_match_mapping: Dict[Type, List[Type]]) -> List[Tuple[Type, str]]:
    """
    Takes a complex type (without any placeholders), gets its return values, and returns productions
    (perhaps each with multiple arguments) that produce the return values.  This method also takes
    care of ``MultiMatchNamedBasicTypes``. If one of the arguments or the return types is a multi
    match type, it gets all the substitutions of those types from ``multi_match_mapping`` and forms
    a list with all possible combinations of substitutions. If the complex type passed to this method
    has no ``MultiMatchNamedBasicTypes``, the returned list will contain a single tuple.  For
    example, if the complex is type ``<a,<<b,c>,d>>``, and ``a`` is a multi match type that matches
    ``e`` and ``f``, this gives the following list of tuples: ``[('d', 'd -> [<a,<<b,c>,d>, e,
    <b,c>]), ('d', 'd -> [<a,<<b,c>,d>, f, <b,c>])]`` Note that we assume there will be no
    productions from the multi match type, and the list above does not contain ``('d', 'd ->
    [<a,<<b,c>,d>, a, <b,c>>]')``.
    """
    return_type = complex_type.return_type()
    if isinstance(return_type, MultiMatchNamedBasicType):
        return_types_matched = list(multi_match_mapping[return_type] if return_type in
                                    multi_match_mapping else return_type.types_to_match)
    else:
        return_types_matched = [return_type]
    arguments = complex_type.argument_types()
    argument_types_matched = []
    for argument_type in arguments:
        if isinstance(argument_type, MultiMatchNamedBasicType):
            matched_types = list(multi_match_mapping[argument_type] if argument_type in
                                 multi_match_mapping else argument_type.types_to_match)
            argument_types_matched.append(matched_types)
        else:
            argument_types_matched.append([argument_type])
    complex_type_productions: List[Tuple[Type, str]] = []
    for matched_return_type in return_types_matched:
        for matched_arguments in itertools.product(*argument_types_matched):
            complex_type_productions.append((matched_return_type,
                                             _make_production_string(return_type,
                                                                     [complex_type] + list(matched_arguments))))
    return complex_type_productions


def get_valid_actions(name_mapping: Dict[str, str],
                      type_signatures: Dict[str, Type],
                      basic_types: Set[Type],
                      multi_match_mapping: Dict[Type, List[Type]] = None,
                      valid_starting_types: Set[Type] = None,
                      num_nested_lambdas: int = 0) -> Dict[str, List[str]]:
    """
    Generates all the valid actions starting from each non-terminal. For terminals of a specific
    type, we simply add a production from the type to the terminal. For all terminal `functions`,
    we additionally add a rule that allows their return type to be generated from an application of
    the function.  For example, the function ``<e,<r,<d,r>>>``, which takes three arguments and
    returns an ``r`` would generate a the production rule ``r -> [<e,<r,<d,r>>>, e, r, d]``.

    For functions that do not contain ANY_TYPE or placeholder types, this is straight-forward.
    When there are ANY_TYPES or placeholders, we substitute the ANY_TYPE with all possible basic
    types, and then produce a similar rule.  For example, the identity function, with type
    ``<#1,#1>`` and basic types ``e`` and ``r``,  would produce the rules ``e -> [<#1,#1>, e]`` and
    ``r -> [<#1,#1>, r]``.

    We additionally add a valid action from the start symbol to all ``valid_starting_types``.

    Parameters
    ----------
    name_mapping : ``Dict[str, str]``
        The mapping of names that appear in your logical form languages to their aliases for NLTK.
        If you are getting all valid actions for a type declaration, this can be the
        ``COMMON_NAME_MAPPING``.
    type_signatures : ``Dict[str, Type]``
        The mapping from name aliases to their types. If you are getting all valid actions for a
        type declaration, this can be the ``COMMON_TYPE_SIGNATURE``.
    basic_types : ``Set[Type]``
        Set of all basic types in the type declaration.
    multi_match_mapping : ``Dict[Type, List[Type]]`` (optional)
        A mapping from `MultiMatchNamedBasicTypes` to the types they can match. This may be
        different from the type's ``types_to_match`` field based on the context. While building action
        sequences that lead to complex types with ``MultiMatchNamedBasicTypes``, if a type does not
        occur in this mapping, the default set of ``types_to_match`` for that type will be used.
    valid_starting_types : ``Set[Type]``, optional
        These are the valid starting types for your grammar; e.g., what types are we allowed to
        parse expressions into?  We will add a "START -> TYPE" rule for each of these types.  If
        this is ``None``, we default to using ``basic_types``.
    num_nested_lambdas : ``int`` (optional)
        Does the language used permit lambda expressions?  And if so, how many nested lambdas do we
        need to worry about?  We'll add rules like "<r,d> -> ['lambda x', d]" for all complex
        types, where the variable is determined by the number of nestings.  We currently only
        permit up to three levels of nesting, just for ease of implementation.
    """
    valid_actions: Dict[str, Set[str]] = defaultdict(set)

    valid_starting_types = valid_starting_types or basic_types
    for type_ in valid_starting_types:
        valid_actions[str(START_TYPE)].add(_make_production_string(START_TYPE, type_))

    complex_types = set()
    for name, alias in name_mapping.items():
        # Lambda functions and variables associated with them get produced in specific contexts. So
        # we do not add them to ``valid_actions`` here, and let ``GrammarState`` deal with it.
        # ``var`` is a special function that some languages (like LambdaDCS) use within lambda
        # functions to indicate the use of a variable (eg.: ``(lambda x (fb:row.row.year (var x)))``)
        # We do not have to produce this function outside the scope of lambda. Even within lambdas,
        # it is a lot easier to not do it, and let the action sequence to logical form transformation
        # logic add it to the output logical forms instead.
        if name in ["lambda", "var", "x", "y", "z"]:
            continue
        name_type = type_signatures[alias]
        # Type to terminal productions.
        for substituted_type in substitute_any_type(name_type, basic_types):
            valid_actions[str(substituted_type)].add(_make_production_string(substituted_type, name))
        # Keeping track of complex types.
        if isinstance(name_type, ComplexType) and name_type != ANY_TYPE:
            complex_types.add(name_type)

    for complex_type in complex_types:
        for substituted_type in substitute_any_type(complex_type, basic_types):
            for head, production in _get_complex_type_production(substituted_type,
                                                                 multi_match_mapping or {}):
                valid_actions[str(head)].add(production)

    # We can produce complex types with a lambda expression, though we'll leave out
    # placeholder types for now.
    for i in range(num_nested_lambdas):
        lambda_var = chr(ord('x') + i)
        # We'll only allow lambdas to be functions that take and return basic types as their
        # arguments, for now.  Also, we're doing this for all possible complex types where
        # the first and second types are basic types. So we may be overgenerating a bit.
        for first_type in basic_types:
            for second_type in basic_types:
                key = ComplexType(first_type, second_type)
                production_string = _make_production_string(key, ['lambda ' + lambda_var, second_type])
                valid_actions[str(key)].add(production_string)

    valid_action_strings = {key: sorted(value) for key, value in valid_actions.items()}
    return valid_action_strings


START_TYPE = NamedBasicType(START_SYMBOL)

# TODO(mattg): We're hard-coding three lambda variables here.  This isn't a great way to do
# this; it's just something that works for now, that we can fix later if / when it's needed.
# If you allow for more than three nested lambdas, or if you want to use different lambda
# variable names, you'll have to change this somehow.
LAMBDA_VARIABLES = set(['x', 'y', 'z'])

def is_nonterminal(production: str) -> bool:
    # TODO(pradeep): This is pretty specific to the assumptions made in converting types to
    # strings (e.g., that we're only using the first letter for types, lowercased).
    # TODO(pradeep): Also we simply check the surface forms here, and this works for
    # wikitables and nlvr. We should ideally let the individual type declarations define their own
    # variants of this method.
    if production in ['<=', '<']:
        # Some grammars (including the wikitables grammar) have "less than" and "less than or
        # equal to" functions that are terminals.  We don't want to treat those like our
        # "<t,d>" types.
        return False
    if production[0] == '<':
        return True
    if production.startswith('fb:'):
        return False
    if len(production) > 1 or production in LAMBDA_VARIABLES:
        return False
    return production[0].islower()
