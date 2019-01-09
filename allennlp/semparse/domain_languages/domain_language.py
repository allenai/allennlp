from collections import defaultdict
from typing import Any, Callable, CallableMeta, Dict, GenericMeta, List, Set, Tuple, Type, Union
import inspect
import logging
import types

from nltk import Tree

from allennlp.common.util import START_SYMBOL
from allennlp.semparse import util

logger = logging.getLogger(__name__)

class PredicateType:
    """
    A base class for `types` in a domain language.  This serves much the same purpose as
    ``typing.Type``, but we add a few conveniences to these types, so we construct separate classes
    for them and group them together under ``PredicateType`` to have a good type annotation for
    these types.
    """
    @staticmethod
    def get_type(type_: Type) -> 'PredicateType':
        if isinstance(type_, CallableMeta):
            callable_args = type_.__args__
            argument_types = [PredicateType.get_type(t) for t in callable_args[:-1]]
            return_type = PredicateType.get_type(callable_args[-1])
            return FunctionType(argument_types, return_type)
        elif isinstance(type_, GenericMeta):
            # This is something like List[int].  type_.__name__ will only give 'List', though, so
            # we need to do some magic here.
            origin = type_.__origin__
            args = type_.__args__
            name = f'{origin.__name__}[{",".join(arg.__name__ for arg in args)}]'
        else:
            name = type_.__name__
        return BasicType(name)

    @staticmethod
    def get_function_type(arg_types: List['PredicateType'], return_type: 'PredicateType') -> 'PredicateType':
        """
        Constructs an NLTK ``ComplexType`` representing a function with the given argument and
        return types.
        """
        # TODO(mattg): We might need to generalize this to just `get_type`, so we can handle
        # functions as arguments correctly in the logic below.
        if not arg_types:
            # Functions with no arguments are basically constants whose type match their return
            # type.
            return return_type
        return FunctionType(arg_types, return_type)


class BasicType(PredicateType):
    """
    A ``PredicateType`` representing a zero-argument predicate (which could technically be a
    function with no arguments or a constant; both are treated the same here).
    """
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.name == other.name
        return NotImplemented


class FunctionType(PredicateType):
    """
    A ``PredicateType`` representing a function with arguments.
    """
    def __init__(self, argument_types: List[PredicateType], return_type: PredicateType) -> None:
        self.argument_types = argument_types
        self.return_type = return_type
        self.name = f'<{",".join(str(arg) for arg in argument_types)}:{return_type}>'

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.name == other.name
        return NotImplemented


class ParsingError(Exception):
    """
    This exception gets raised when there is a parsing error during logical form processing.  This
    might happen because you're not handling the full set of possible logical forms, for instance,
    and having this error provides a consistent way to catch those errors and log how frequently
    this occurs.
    """
    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class ExecutionError(Exception):
    """
    This exception gets raised when you're trying to execute a logical form that your executor does
    not understand. This may be because your logical form contains a function with an invalid name
    or a set of arguments whose types do not match those that the function expects.
    """
    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def predicate(function: Callable) -> Callable:  # pylint: disable=invalid-name
    """
    This is intended to be used as a decorator when you are implementing your ``DomainLanguage``.
    This marks a function on a ``DomainLanguage`` subclass as a predicate that can be used in the
    language.  See the :class:`DomainLanguage` docstring for an example usage, and for what using
    this does.
    """
    setattr(function, '_is_predicate', True)
    return function


def nltk_tree_to_logical_form(tree: Tree) -> str:
    """
    Given an ``nltk.Tree`` representing the syntax tree that generates a logical form, this method
    produces the actual (lisp-like) logical form, with all of the non-terminal symbols converted
    into the correct number of parentheses.

    This is used in the logic that converts action sequences back into logical forms.  It's very
    unlikely that you will need this anywhere else.
    """
    # nltk.Tree actually inherits from `list`, so you use `len()` to get the number of children.
    # We're going to be explicit about checking length, instead of using `if tree:`, just to avoid
    # any funny business nltk might have done (e.g., it's really odd if `if tree:` evaluates to
    # `False` if there's a single leaf node with no children).
    if len(tree) == 0:  # pylint: disable=len-as-condition
        return tree.label()
    if len(tree) == 1:
        return tree[0].label()
    return '(' + ' '.join(nltk_tree_to_logical_form(child) for child in tree) + ')'


class DomainLanguage:
    """
    A ``DomainLanguage`` specifies the functions available to use for a semantic parsing task.  You
    write execution code for these functions, and we will automatically induce a grammar from those
    functions and give you a lisp interpreter that can use those functions.  For example:

    .. code-block:: python

        class Arithmetic(DomainLanguage):
            @predicate
            def add(self, num1: int, num2: int) -> int:
                return num1 + num2

            @predicate
            def halve(self, num: int) -> int:
                return num / 2

            ...

    Instantiating this class now gives you a language object that can parse and execute logical
    forms, can convert logical forms to action sequences (linearized abstract syntax trees) and
    back again, and can list all valid production rules in a grammar induced from the specified
    functions.

    .. code-block:: python

        >>> l = Arithmetic()
        >>> l.execute("(add 2 3)")
        5
        >>> l.execute("(halve (add 12 4))")
        8
        >>> l.logical_form_to_action_sequence("(add 2 3)")
        # See the docstring for this function for an description of what these strings mean.
        ['@start@ -> int', 'int -> [<int,int:int>, int, int]', '<int,int:int> -> add',
         'int -> 2', 'int -> 3']
        >>> l.action_sequence_to_logical_form(l.logical_form_to_action_sequence('(add 2 3)'))
        '(add 2 3)'
        >>> l.get_nonterminal_productions()
        {'<int,int:int>': ['add', 'divide', 'multiply', 'subtract'], '<int:int>': ['halve'], ...}

    This is done with some reflection magic, with the help of the ``@predicate`` decorator and type
    annotations.  For a method you define on a ``DomainLanguage`` subclass to be included in the
    language, it *must* be decorated with ``@predicate``, and it *must* have type annotations on
    all arguments and on its return type.  You can also add predicates and constants to the
    language using the :func:`add_predicate` and :func:`add_constant` functions, if you choose
    (minor point: constants with generic types (like ``Set[int]``) must currently be specified as
    predicates, as the ``allowed_constants`` dictionary doesn't pass along the generic type
    information).

    The language we construct is purely functional - no defining variables or using lambda
    functions, or anything like that.  If you would like to extend this code to handle more complex
    languages, open an issue on github.
    """
    def __init__(self,
                 allowed_constants: Dict[str, Any] = None,
                 start_types: Set[Type] = None) -> None:
        self._functions: Dict[str, Callable] = {}
        self._function_types: Dict[str, PredicateType] = {}
        self._start_types: Set[PredicateType] = set([PredicateType.get_type(type_) for type_ in start_types])
        for name in dir(self):
            if isinstance(getattr(self, name), types.MethodType):
                function = getattr(self, name)
                if getattr(function, '_is_predicate', False):
                    self.add_predicate(name, function)
        for name, value in allowed_constants.items():
            self.add_constant(name, value)
        # Caching this to avoid recomputing it every time `get_nonterminal_productions` is called.
        self._nonterminal_productions: Dict[str, List[str]] = None

    def execute(self, logical_form: str):
        """Executes a logical form, using whatever predicates you have defined."""
        if not hasattr(self, '_functions'):
            raise RuntimeError("You must call super().__init__() in your Language constructor")
        logical_form = logical_form.replace(",", " ")
        expression = util.lisp_to_nested_expression(logical_form)
        return self._execute_expression(expression)

    def get_nonterminal_productions(self) -> Dict[str, List[str]]:
        """
        Induces a grammar from the defined collection of predicates in this language and returns
        all productions in that grammar, keyed by the non-terminal they are expanding.

        This includes terminal productions implied by each predicate as well as productions for the
        `return type` of each defined predicate.  For example, defining a "multiply" predicate adds
        a "<int,int:int> -> multiply" terminal production to the grammar, and `also` a "int ->
        [<int,int:int>, int, int]" non-terminal production, because I can use the "multiply"
        predicate to produce an int.
        """
        if not self._nonterminal_productions:
            actions: Dict[str, List[str]] = defaultdict(list)
            # If you didn't give us a set of valid start types, we'll assume all types we know
            # about (including functional types) are valid start types.
            start_types = self._start_types or set(self._function_types.values())
            for start_type in start_types:
                actions[START_SYMBOL].append(f"{START_SYMBOL} -> {start_type}")
            for name, function_type in self._function_types.items():
                actions[str(function_type)].append(f"{function_type} -> {name}")
                if isinstance(function_type, FunctionType):
                    return_type = function_type.return_type
                    arg_types = function_type.argument_types
                    right_side = f"[{function_type}, {', '.join(str(arg_type) for arg_type in arg_types)}]"
                    actions[str(return_type)].append(f"{return_type} -> {right_side}")
            self._nonterminal_productions = dict(actions)
        return self._nonterminal_productions

    def all_possible_productions(self) -> List[str]:
        """
        Returns a sorted list of all production rules in the grammar induced by
        :func:`get_nonterminal_productions`.
        """
        all_actions = set()
        for action_set in self.get_nonterminal_productions().values():
            all_actions.update(action_set)
        return sorted(all_actions)

    def logical_form_to_action_sequence(self, logical_form: str) -> List[str]:
        """
        Converts a logical form into a linearization of the production rules from its abstract
        syntax tree.  The linearization is top-down, depth-first.

        Each production rule is formatted as "LHS -> RHS", where "LHS" is a single non-terminal
        type, and RHS is either a terminal or a list of non-terminals (other possible values for
        RHS in a more general context-free grammar are not produced by our grammar induction
        logic).

        Non-terminals are `types` in the grammar, either basic types (like ``int``, ``str``, or
        some class that you define), or functional types, represented with angle brackets with a
        colon separating arguments from the return type.  Multi-argument functions have commas
        separating their argument types.  For example, ``<int:int>`` is a function that takes an
        integer and returns an integer, and ``<int,int:int>`` is a function that takes two integer
        arguments and returns an integer.

        As an example translation from logical form to complete action sequence, the logical form
        ``(add 2 3)`` would be translated to ``['@start@ -> int', 'int -> [<int,int:int>, int, int]',
        '<int,int:int> -> add', 'int -> 2', 'int -> 3']``.
        """
        expression = util.lisp_to_nested_expression(logical_form)
        try:
            transitions, start_type = self._get_transitions(expression, expected_type=None)
            if self._start_types and start_type not in self._start_types:
                raise ParsingError(f"Expression had unallowed start type of {start_type}: {expression}")
        except ParsingError:
            logger.error(f'Error parsing logical form: {logical_form}')
            raise
        transitions.insert(0, f'@start@ -> {start_type}')
        return transitions

    def action_sequence_to_logical_form(self, action_sequence: List[str]) -> str:
        """
        Takes an action sequence as produced by :func:`logical_form_to_action_sequence`, which is a
        linearization of an abstract syntax tree, and reconstructs the logical form defined by that
        abstract syntax tree.
        """
        # Basic outline: we assume that the bracketing that we get in the RHS of each action is the
        # correct bracketing for reconstructing the logical form.  This is true when there is no
        # currying in the action sequence.  Given this assumption, we just need to construct a tree
        # from the action sequence, then output all of the leaves in the tree, with brackets around
        # the children of all non-terminal nodes.

        remaining_actions = [action.split(" -> ") for action in action_sequence]
        tree = Tree(remaining_actions[0][1], [])

        try:
            remaining_actions = self._construct_node_from_actions(tree, remaining_actions[1:])
        except ParsingError:
            logger.error("Error parsing action sequence: %s", action_sequence)
            raise

        if remaining_actions:
            logger.error("Error parsing action sequence: %s", action_sequence)
            logger.error("Remaining actions were: %s", remaining_actions)
            raise ParsingError("Extra actions in action sequence")
        return nltk_tree_to_logical_form(tree)

    def add_predicate(self, name: str, function: Callable):
        """
        Adds a predicate to this domain language.  Typically you do this with the ``@predicate``
        decorator on the methods in your class.  But, if you need to for whatever reason, you can
        also call this function yourself with a (type-annotated) function to add it to your
        language.
        """
        signature = inspect.signature(function)
        argument_types = [param.annotation for param in signature.parameters.values()]
        return_type = signature.return_annotation
        # TODO(mattg): this might need to just call PredicateType.get_type, or something - what if
        # one of these is a function?
        argument_nltk_types: List[PredicateType] = [PredicateType.get_type(arg_type)
                                                    for arg_type in argument_types]
        return_nltk_type = PredicateType.get_type(return_type)
        function_nltk_type = PredicateType.get_function_type(argument_nltk_types, return_nltk_type)
        self._functions[name] = function
        self._function_types[name] = function_nltk_type

    def add_constant(self, name: str, value: Any):
        """
        Adds a constant to this domain language.  You would typically just pass in a list of
        constants to the ``super().__init__()`` call in your constructor, but you can also call
        this method to add constants if it is more convenient.

        Because we construct a grammar over this language for you, in order for the grammar to be
        finite we cannot allow arbitrary constants.  Having a finite grammar is important when
        you're doing semantic parsing - we need to be able to search over this space, and compute
        normalized probability distributions.
        """
        constant_type = PredicateType.get_type(type(value))
        self._functions[name] = lambda: value
        self._function_types[name] = constant_type

    def _execute_expression(self, expression: Any):
        """
        This does the bulk of the work of executing a logical form, recursively executing a single
        expression.  Basically, if the expression is a function we know about, we evaluate its
        arguments then call the function.  If it's a list, we evaluate all elements of the list.
        If it's a constant (or a zero-argument function), we evaluate the constant.
        """
        # pylint: disable=too-many-return-statements
        if isinstance(expression, list):
            if isinstance(expression[0], list):
                function = self._execute_expression(expression[0])
            elif expression[0] in self._functions:
                function = self._functions[expression[0]]
            else:
                if isinstance(expression[0], str):
                    raise ExecutionError(f"Unrecognized function: {expression[0]}")
                else:
                    raise ExecutionError(f"Unsupported expression type: {expression}")
            arguments = [self._execute_expression(arg) for arg in expression[1:]]
            try:
                return function(*arguments)
            except (TypeError, ValueError) as error:
                raise ExecutionError(f"Error executing expression {expression}: {error}")
        elif isinstance(expression, str):
            if expression not in self._functions:
                raise ExecutionError(f"Unrecognized constant: {expression}")
            # This is a bit of a quirk in how we represent constants and zero-argument functions.
            # For consistency, constants are wrapped in a zero-argument lambda.  So both constants
            # and zero-argument functions are callable in `self._functions`, and are `BasicTypes`
            # in `self._function_types`.  For these, we want to return
            # `self._functions[expression]()` _calling_ the zero-argument function.  If we get a
            # `FunctionType` in here, that means we're referring to the function as a first-class
            # object, instead of calling it (maybe as an argument to a higher-order function).  In
            # that case, we return the function _without_ calling it.
            if isinstance(self._function_types[expression], FunctionType):
                return self._functions[expression]
            else:
                return self._functions[expression]()
            return self._functions[expression]
        else:
            raise ExecutionError("Not sure how you got here. Please open a github issue with details.")

    def _get_transitions(self, expression: Any, expected_type: PredicateType) -> Tuple[List[str], PredicateType]:
        """
        This is used when converting a logical form into an action sequence.  This piece
        recursively translates a lisp expression into an action sequence, making sure we match the
        expected type (or using the expected type to get the right type for constant expressions).
        """
        if isinstance(expression, (list, tuple)):
            function_transitions, return_type, argument_types = self._get_function_transitions(expression[0],
                                                                                               expected_type)
            if len(argument_types) != len(expression[1:]):
                raise ParsingError(f'Wrong number of arguments for function in {expression}')
            argument_transitions = []
            for argument_type, subexpression in zip(argument_types, expression[1:]):
                argument_transitions.extend(self._get_transitions(subexpression, argument_type)[0])
            return function_transitions + argument_transitions, return_type
        elif isinstance(expression, str):
            if expression not in self._functions:
                raise ParsingError(f"Unrecognized constant: {expression}")
            constant_type = self._function_types[expression]
            if expected_type and expected_type != constant_type:
                raise ParsingError(f'{expression} did not have expected type {expected_type} '
                                   f'(found {constant_type})')
            return [f'{constant_type} -> {expression}'], constant_type
        else:
            raise ParsingError('Not sure how you got here. Please open an issue on github with details.')

    def _get_function_transitions(self,
                                  expression: Union[str, List],
                                  expected_type: PredicateType) -> Tuple[List[str],
                                                                         PredicateType,
                                                                         List[PredicateType]]:
        """
        A helper method for ``_get_transitions``.  This gets the transitions for the predicate
        itself in a function call.  If we only had simple functions (e.g., "(add 2 3)"), this would
        be pretty straightforward and we wouldn't need a separate method to handle it.  We split it
        out into its own method because handling higher-order functions is complicated (e.g.,
        something like "((negate add) 2 3)").
        """
        # This first block handles getting the transitions and function type (and some error
        # checking) _just for the function itself_.  If this is a simple function, this is easy; if
        # it's a higher-order function, it involves some recursion.
        if isinstance(expression, list):
            # This is a higher-order function.  TODO(mattg): we'll just ignore type checking on
            # higher-order functions, for now.
            transitions, function_type = self._get_transitions(expression, None)
        elif expression in self._functions:
            name = expression
            function_type = self._function_types[expression]
            transitions = [f'{function_type} -> {name}']
        else:
            if isinstance(expression, str):
                raise ParsingError(f"Unrecognized function: {expression[0]}")
            else:
                raise ParsingError(f"Unsupported expression type: {expression}")
        if not isinstance(function_type, FunctionType):
            raise ParsingError(f'Zero-arg function or constant called with arguments: {name}')

        # Now that we have the transitions for the function itself, and the function's type, we can
        # get argument types and do the rest of the transitions.
        argument_types = function_type.argument_types
        return_type = function_type.return_type
        right_side = f'[{function_type}, {", ".join(str(arg) for arg in argument_types)}]'
        first_transition = f'{return_type} -> {right_side}'
        transitions.insert(0, first_transition)
        if expected_type and expected_type != return_type:
            raise ParsingError(f'{expression} did not have expected type {expected_type} '
                               f'(found {return_type})')
        return transitions, return_type, argument_types

    def _construct_node_from_actions(self,
                                     current_node: Tree,
                                     remaining_actions: List[List[str]]) -> List[List[str]]:
        """
        Given a current node in the logical form tree, and a list of actions in an action sequence,
        this method fills in the children of the current node from the action sequence, then
        returns whatever actions are left.

        For example, we could get a node with type ``c``, and an action sequence that begins with
        ``c -> [<r,c>, r]``.  This method will add two children to the input node, consuming
        actions from the action sequence for nodes of type ``<r,c>`` (and all of its children,
        recursively) and ``r`` (and all of its children, recursively).  This method assumes that
        action sequences are produced `depth-first`, so all actions for the subtree under ``<r,c>``
        appear before actions for the subtree under ``r``.  If there are any actions in the action
        sequence after the ``<r,c>`` and ``r`` subtrees have terminated in leaf nodes, they will be
        returned.
        """
        if not remaining_actions:
            logger.error("No actions left to construct current node: %s", current_node)
            raise ParsingError("Incomplete action sequence")
        left_side, right_side = remaining_actions.pop(0)
        if left_side != current_node.label():
            logger.error("Current node: %s", current_node)
            logger.error("Next action: %s -> %s", left_side, right_side)
            logger.error("Remaining actions were: %s", remaining_actions)
            raise ParsingError("Current node does not match next action")
        if right_side[0] == '[':
            # This is a non-terminal expansion, with more than one child node.
            for child_type in right_side[1:-1].split(', '):
                child_node = Tree(child_type, [])
                current_node.append(child_node)  # you add a child to an nltk.Tree with `append`
                # For now, we assume that all children in a list like this are non-terminals, so we
                # recurse on them.  I'm pretty sure that will always be true for the way our
                # grammar induction works.  We can revisit this later if we need to.
                remaining_actions = self._construct_node_from_actions(child_node, remaining_actions)
        else:
            # The current node is a pre-terminal; we'll add a single terminal child.  By
            # construction, the right-hand side of our production rules are only ever terminal
            # productions or lists of non-terminals.
            current_node.append(Tree(right_side, []))  # you add a child to an nltk.Tree with `append`
        return remaining_actions
