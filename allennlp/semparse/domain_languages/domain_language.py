from typing import Any, Callable, Dict, GenericMeta, List, Set, Tuple, Type
import inspect
import logging
import types

from nltk import Tree
from nltk.sem.logic import Type as NltkType

from allennlp.semparse import util
from allennlp.semparse.type_declarations import type_declaration
from allennlp.semparse.type_declarations.type_declaration import ComplexType, NamedBasicType, NameMapper

logger = logging.getLogger(__name__)


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
    function.is_predicate = True  # type: ignore
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
        ['@start@ -> i', 'i -> [<i,<i,i>>, i, i]', '<i,<i,i>> -> add', 'i -> 2', 'i -> 3']
        >>> l.action_sequence_to_logical_form(l.logical_form_to_action_sequence('(add 2 3)'))
        '(add 2 3)'
        >>> l.get_valid_actions()
        {'<i,<i,i>>': ['add', 'divide', 'multiply', 'subtract'], '<i,i>': ['halve'], ...}

    This is done with some reflection magic, with the help of the ``@predicate`` decorator and type
    annotations.  For a predicate to be included in the language, it *must* be decorated with
    ``@predicate``, and it *must* have type annotations on all arguments and on its return type.

    For type annotations, we currently support ``int``, ``float``, ``str``, and similar
    non-collection built in types, as well as any non-generic type you define yourself.  Generic
    types (things like ``List[str]``) are not currently supported very well - you will likely be
    able to execute logical forms that use them, but the grammar induction won't know how to create
    a ``List[str]``, and so we also won't produce a correct action sequence for them.

    The language we construct is also purely functional - no defining variables or using lambda
    functions, or anything like that.  If you would like to extend this code to handle more complex
    languages, open an issue on github.
    """
    def __init__(self):
        self._type_map: Dict[Type, NltkType] = {}
        self._functions: Dict[str, Callable] = {}
        self._function_types: Dict[str, Tuple[List[NltkType], NltkType]] = {}
        self._name_mapper = NameMapper()
        self.start_types: Set[Type] = None
        for name in dir(self):
            if isinstance(getattr(self, name), types.MethodType):
                function = getattr(self, name)
                if hasattr(function, 'is_predicate') and function.is_predicate:
                    self.add_predicate(name, function)
        # Caching this to avoid recompting it every time `get_valid_actions` is called.
        self._valid_actions: Dict[str, List[str]] = None

    def execute(self, logical_form: str):
        """Executes a logical form, using whatever predicates you have defined."""
        logical_form = logical_form.replace(",", " ")
        expression = util.lisp_to_nested_expression(logical_form)
        return self._execute_expression(expression)

    def get_valid_actions(self) -> Dict[str, List[str]]:
        """
        Induces a grammar from the defined collection of predicates in this language.  This
        includes terminal productions implied by each predicate as well as productions for the
        `return type` of each defined predicate.  For example, defining a "multiply" predicate adds
        a "<i,<i,i>> -> multiplty" terminal production to the grammar, and `also` a "i ->
        [<i,<i,i>>, i, i]" non-terminal production, because I can use the "multiply" predicate to
        produce an integer.
        """
        if not self._valid_actions:
            basic_types = set([nltk_type for nltk_type in self._type_map.values()
                               if isinstance(nltk_type, NamedBasicType)])
            if self.start_types:
                # Not sure why pylint misses this...
                # pylint: disable=not-an-iterable
                start_types = set([self._get_basic_nltk_type(type_) for type_ in self.start_types])
                # pylint: enable=not-an-iterable
            else:
                start_types = None
            self._valid_actions = type_declaration.get_valid_actions(self._name_mapper.name_mapping,
                                                                     self._name_mapper.type_signatures,
                                                                     basic_types,
                                                                     valid_starting_types=start_types)
        return self._valid_actions

    def logical_form_to_action_sequence(self, logical_form: str) -> List[str]:
        """
        Converts a logical form into a linearization of the production rules from its abstract
        syntax tree.  For example, the logical form ``(add 2 3)`` would be translated to something
        like ``['i -> [<i,<i,i>>, i, i]', '<i,<i,i>> -> add', 'i -> 2', 'i -> 3']``.  The
        linearization is top-down, depth-first.
        """
        expression = util.lisp_to_nested_expression(logical_form)
        try:
            transitions = self._get_transitions(expression, expected_type=None)
        except ParsingError:
            logger.error(f'Error parsing logical form: {logical_form}')
            raise
        # TODO(mattg): we _could_ check that the start_type here is one of self.start_types.
        start_type = transitions[0].split(' -> ')[0]
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
        self._functions[name] = function
        signature = inspect.signature(function)
        argument_types = [param.annotation for param in signature.parameters.values()]
        return_type = signature.return_annotation
        argument_nltk_types = [self._get_basic_nltk_type(arg_type) for arg_type in argument_types]
        return_nltk_type = self._get_basic_nltk_type(return_type)
        function_nltk_type = self._get_function_type(argument_nltk_types, return_nltk_type)
        self._function_types[name] = (argument_nltk_types, return_nltk_type)
        self._name_mapper.map_name_with_signature(name, function_nltk_type)

    def _get_basic_nltk_type(self, type_: Type) -> NltkType:
        """
        Constructs an NLTK ``NamedBasicType`` representing the given type.  This is typically a
        simple class, like int, string, Point, or Box.  It could also be a Tuple[int, string] or a
        List[str].  This is `not` for functional types, however.
        """
        if type_ not in self._type_map:
            if isinstance(type_, GenericMeta):
                # This is something like List[int].  type_.__name__ will only give 'List', though, so
                # we need to do some magic here.
                origin = type_.__origin__
                args = type_.__args__
                name = '_'.join([origin.__name__] + [arg.__name__ for arg in args])
                # TODO(mattg): we probably need to add a grammar rule in here too, somehow, so we
                # can actually _produce_ a list of things.
            else:
                name = type_.__name__
            self._type_map[type_] = NamedBasicType(name.upper())
        return self._type_map[type_]

    @staticmethod
    def _get_function_type(arg_types: List[NltkType], return_type: NltkType) -> NltkType:
        """
        Constructs an NLTK ``ComplexType`` representing a function with the given argument and
        return types.
        """
        if not arg_types:
            # Functions with no arguments are basically constants whose type match their return
            # type.
            return return_type

        # NLTK's logic classes _curry_ multi-argument functions.  A `ComplexType` is a
        # single-argument function, and you nest these for multi-argument functions.  This for loop
        # does the nesting for multi-argument functions.
        right_argument = return_type
        for arg_type in reversed(arg_types):
            final_type = ComplexType(arg_type, right_argument)
            right_argument = final_type
        return final_type

    def _is_terminal(self, name: str) -> bool:
        """
        This is used to know when we should recurse when converting action sequences to logical
        forms.  If a piece of the right-hand-side of a production rule is a terminal, we don't
        recurse on it.
        """
        if name in self._functions:
            return True
        if name[0] == '"' and name[-1] == "'":
            return True
        if name[0] == "'" and name[-1] == '"':
            return True
        try:
            int(name)
            return True
        except ValueError:
            pass
        try:
            float(name)
            return True
        except ValueError:
            pass
        return False

    def _execute_expression(self, expression: Any):
        """
        This does the bulk of the work of executing a logical form, recursively executing a single
        expression.  Basically, if the expression is a function we know about, we evaluate its
        arguments then call the function.  If it's a list, we evaluate all elements of the list.
        If it's a constant (or a zero-argument function), we evaluate the constant.
        """
        # pylint: disable=too-many-return-statements
        if isinstance(expression, (list, tuple)):
            if expression[0] in self._functions:
                function = self._functions[expression[0]]
                arguments = [self._execute_expression(arg) for arg in expression[1:]]
                try:
                    return function(*arguments)
                except (TypeError, ValueError) as error:
                    raise ExecutionError(f"Error executing expression {expression}: {error}")
            else:
                return [self._execute_expression(item) for item in expression]
        elif isinstance(expression, str):
            if expression[0] == '"' and expression[-1] == '"':
                return expression[1:-1]
            if expression[0] == "'" and expression[-1] == "'":
                return expression[1:-1]
            if expression in self._functions:
                try:
                    return self._functions[expression]()
                except (TypeError, ValueError) as error:
                    raise ExecutionError(f"Error executing expression {expression}: {error}")
            try:
                int_value = int(expression)
                return int_value
            except ValueError:
                pass
            try:
                float_value = float(expression)
                return float_value
            except ValueError:
                pass
        return expression

    def _get_transitions(self, expression: Any, expected_type: NltkType) -> List[str]:
        """
        This is used when converting a logical form into an action sequence.  This piece
        recursively translates a lisp expression into an action sequence, making sure we match the
        expected type (or using the expected type to get the right type for constant expressions).
        """
        if isinstance(expression, (list, tuple)):
            if expression[0] in self._functions:
                name = expression[0]
                argument_types, return_type = self._function_types[name]
                function_type = self._name_mapper.get_signature(name)
                if expected_type and expected_type != return_type:
                    raise ParsingError(f'{expression} did not have expected type {expected_type}')
                right_side = f'[{function_type}, {", ".join(str(arg) for arg in argument_types)}]'
                first_transition = f'{return_type} -> {right_side}'
                second_transition = f'{function_type} -> {name}'
                argument_transitions = []
                if len(argument_types) != len(expression[1:]):
                    raise ParsingError(f'Wrong number of arguments for function in {expression}')
                for argument_type, subexpression in zip(argument_types, expression[1:]):
                    argument_transitions.extend(self._get_transitions(subexpression, argument_type))
                return [first_transition, second_transition] + argument_transitions
            else:
                raise ParsingError('Bare lists not implemented yet')
        elif isinstance(expression, str):
            # TODO(mattg): do type inference on constants, and check for matches (and allow
            # top-level constant expressions).
            # TODO(mattg): this could also possibly cause problems with inconsistent handling of
            # quotation marks.  We should probably pull the constant handling out into its own
            # function in both `_get_transitions` and `_execute_expression`.
            if expected_type:
                return [f'{expected_type} -> {expression}']
            else:
                raise ParsingError('Constant expressions not implemented yet')
        else:
            raise ParsingError('Not sure how you got here.  Raise an issue on github with details')

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
                if not self._is_terminal(child_type):
                    remaining_actions = self._construct_node_from_actions(child_node,
                                                                          remaining_actions)
        elif self._is_terminal(right_side):
            # The current node is a pre-terminal; we'll add a single terminal child.
            current_node.append(Tree(right_side, []))  # you add a child to an nltk.Tree with `append`
        else:
            # The only way this can happen is if you have a unary non-terminal production rule.
            # That is almost certainly not what you want with this kind of grammar, so we'll crash.
            # If you really do want this, open a PR with a valid use case.
            raise ParsingError(f"Found a unary production rule: {left_side} -> {right_side}. "
                               "Are you sure you want a unary production rule in your grammar?")
        return remaining_actions
