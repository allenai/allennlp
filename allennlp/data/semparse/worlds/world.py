from typing import List, Dict, Set, Tuple
from collections import defaultdict
import logging
import re

from nltk.sem.logic import Expression, LambdaExpression, BasicType, Type

from allennlp.data.semparse.type_declarations import type_declaration as types
from allennlp.data.semparse import util as semparse_util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ParsingError(Exception):
    """
    This exception gets raised when there is a parsing error during logical form processing.  This
    might happen because you're not handling the full set of possible logical forms, for instance,
    and having this error provides a consistent way to catch those errors and log how frequently
    this occurs.
    """
    def __init__(self, message):
        super(ParsingError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class World:
    """
    Base class for defining a world in a new domain. This class defines a method to translate a
    logical form as per a naming convention that works with NLTK's ``LogicParser``. The sub-classes
    can decide on the convention by overriding the ``_map_name`` method that does token level
    mapping. This class also defines methods for transforming logical form strings into parsed
    ``Expressions``, and ``Expressions`` into action sequences.

    Parameters
    ----------
    constant_type_prefixes : ``Dict[str, BasicType]`` (optional)
        If you have an unbounded number of constants in your domain, you are required to add
        prefixes to their names to denote their types. This is the mapping from prefixes to types.
    global_type_signatures : ``Dict[str, Type]`` (optional)
        A mapping from translated names to their types.
    global_name_mapping : ``Dict[str, str]`` (optional)
        A name mapping from the original names in the domain to the translated names.
    num_nested_lambdas : ``int`` (optional)
        Does the language used in this ``World`` permit lambda expressions?  And if so, how many
        nested lambdas do we need to worry about?  This is important when considering the space of
        all possible actions, which we need to enumerate a priori for the parser.
    """
    def __init__(self,
                 constant_type_prefixes: Dict[str, BasicType] = None,
                 global_type_signatures: Dict[str, Type] = None,
                 global_name_mapping: Dict[str, str] = None,
                 num_nested_lambdas: int = 0) -> None:
        # NLTK has a naming convention for variable types. If the world has predicate or entity names beyond
        # what's defined in the COMMON_NAME_MAPPING, they need to be added to this dict.
        # We initialize this dict with common predicate names and update it as we process logical forms.
        self.local_name_mapping: Dict[str, str] = {}
        # Similarly, these are the type signatures not in the COMMON_TYPE_SIGNATURE.
        self.local_type_signatures: Dict[str, Type] = {}
        self.global_name_mapping = global_name_mapping or {}
        self.global_type_signatures = global_type_signatures or {}
        # We keep a reverse map as well to put the terminals back in action sequences.
        self.reverse_name_mapping = {mapped_name: name for name, mapped_name in self.global_name_mapping.items()}
        type_prefixes = constant_type_prefixes or {}
        self._num_nested_lambdas = num_nested_lambdas
        if num_nested_lambdas > 3:
            raise NotImplementedError("For ease of implementation, we currently only handle at "
                                      "most three nested lambda expressions")
        self._logic_parser = types.DynamicTypeLogicParser(constant_type_prefixes=type_prefixes,
                                                          type_signatures=self.global_type_signatures)
        self._right_side_indexed_actions: Dict[str, List[Tuple[str, str]]] = None

    def get_name_mapping(self) -> Dict[str, str]:
        # Python 3.5 syntax for merging two dictionaries.
        return {**self.global_name_mapping, **self.local_name_mapping}

    def get_type_signatures(self) -> Dict[str, str]:
        # Python 3.5 syntax for merging two dictionaries.
        return {**self.global_type_signatures, **self.local_type_signatures}

    def get_valid_actions(self) -> Dict[str, List[str]]:
        return types.get_valid_actions(self.get_name_mapping(),
                                       self.get_type_signatures(),
                                       self.get_basic_types(),
                                       valid_starting_types=self.get_valid_starting_types(),
                                       num_nested_lambdas=self._num_nested_lambdas)

    def get_paths_to_root(self,
                          action: str,
                          max_path_length: int = 20,
                          beam_size: int = 30,
                          max_num_paths: int = 10) -> List[List[str]]:
        """
        For a given action, returns at most ``max_num_paths`` paths to the root (production with
        ``@START@``) that are not longer than ``max_path_length``.
        """
        action_left_side, _ = action.split(' -> ')
        right_side_indexed_actions = self._get_right_side_indexed_actions()
        lists_to_expand: List[Tuple[str, List[str]]] = [(action_left_side, [action])]
        completed_paths = []
        while lists_to_expand:
            need_to_expand = False
            for left_side, path in lists_to_expand:
                if left_side == types.START_SYMBOL:
                    completed_paths.append(path)
                else:
                    need_to_expand = True
            if not need_to_expand or len(completed_paths) >= max_num_paths:
                break
            # We keep track of finished and unfinished lists separately because we truncate the beam
            # later, and we want the finished lists to be at the top of the beam.
            finished_new_lists = []
            unfinished_new_lists = []
            for left_side, actions in lists_to_expand:
                for next_left_side, next_action in right_side_indexed_actions[left_side]:
                    if next_action in actions:
                        # Ignoring paths with loops (of size 1)
                        continue
                    new_actions = list(actions)
                    new_actions.append(next_action)
                    # Ignoring lists that are too long, and have too many repetitions.
                    path_length = len(new_actions)
                    if path_length <= max_path_length or next_left_side == types.START_SYMBOL:
                        if next_left_side == types.START_SYMBOL:
                            finished_new_lists.append((next_left_side, new_actions))
                        else:
                            unfinished_new_lists.append((next_left_side, new_actions))
            new_lists = finished_new_lists + unfinished_new_lists
            lists_to_expand = new_lists[:beam_size]
        return completed_paths[:max_num_paths]

    def all_possible_actions(self) -> List[str]:
        all_actions = set()
        for action_set in self.get_valid_actions().values():
            all_actions.update(action_set)
        for i in range(self._num_nested_lambdas):
            lambda_var = chr(ord('x') + i)
            for basic_type in self.get_basic_types():
                production = f"{basic_type} -> {lambda_var}"
                all_actions.add(production)
        return sorted(all_actions)

    def _get_right_side_indexed_actions(self):
        if not self._right_side_indexed_actions:
            self._right_side_indexed_actions = defaultdict(list)
            all_actions = self.all_possible_actions()
            for possible_action in all_actions:
                left_side, right_side = possible_action.split(' -> ')
                if '[' not in right_side:
                    self._right_side_indexed_actions[right_side].append((left_side, possible_action))
                else:
                    right_side_parts = right_side[1:-1].split(', ')
                    for right_side_part in right_side_parts:
                        self._right_side_indexed_actions[right_side_part].append((left_side,
                                                                                  possible_action))
        return self._right_side_indexed_actions

    def get_basic_types(self) -> Set[Type]:
        """
        Returns the set of basic types (types of entities) in the world.
        """
        raise NotImplementedError

    def get_valid_starting_types(self) -> Set[Type]:
        """
        Returns the set of all types t, such that actions ``@START@ -> t`` are valid. In other
        words, these are all the possible types of complete logical forms in this world.
        """
        raise NotImplementedError

    def parse_logical_form(self,
                           logical_form: str,
                           remove_var_function: bool = True) -> Expression:
        """
        Takes a logical form as a string, maps its tokens using the mapping and returns a parsed expression.

        Parameters
        ----------
        logical_form : ``str``
            Logical form to parse
        remove_var_function : ``bool`` (optional)
            ``var`` is a special function that some languages use within lambda founctions to
            indicate the usage of a variable. If your language uses it, and you do not want to
            include it in the parsed expression, set this flag. You may want to do this if you are
            generating an action sequence from this parsed expression, because it is easier to let
            the decoder not produce this function due to the way constrained decoding is currently
            implemented.
        """
        if not logical_form.startswith("("):
            logical_form = "(%s)" % logical_form
        if remove_var_function:
            # Replace "(x)" with "x"
            logical_form = re.sub(r'\(([x-z])\)', r'\1', logical_form)
            # Replace "(var x)" with "(x)"
            logical_form = re.sub(r'\(var ([x-z])\)', r'(\1)', logical_form)
        parsed_lisp = semparse_util.lisp_to_nested_expression(logical_form)
        translated_string = self._process_nested_expression(parsed_lisp)
        type_signature = self.local_type_signatures.copy()
        type_signature.update(self.global_type_signatures)
        return self._logic_parser.parse(translated_string, signature=type_signature)

    def get_action_sequence(self, expression: Expression) -> List[str]:
        """
        Returns the sequence of actions (as strings) that resulted in the given expression.
        """
        # Starting with the type of the whole expression
        return self._get_transitions(expression,
                                     ["%s -> %s" % (types.START_TYPE, expression.type)])

    def get_logical_form(self,
                         action_sequence: List[str],
                         add_var_function: bool = True) -> str:
        """
        Takes an action sequence and constructs a logical form from it. This is useful if you want
        to get a logical form from a decoded sequence of actions generated by a transition based
        semantic parser.

        Parameters
        ----------
        action_sequence : ``List[str]``
            The sequence of actions as strings (eg.: ``['@START@ -> t', 't -> <e,t>', ...]``).
        add_var_function : ``bool`` (optional)
             ``var`` is a special function that some languages use within lambda functions to
             indicate the use of a variable (eg.: ``(lambda x (fb:row.row.year (var x)))``). Due to
             the way constrained decoding is currently implemented, it is easier for the decoder to
             not produce these functions. In that case, setting this flag adds the function in the
             logical form even though it is not present in the action sequence.
        """
        split_actions = [action.split(" -> ") for action in action_sequence]
        terminals: List[Tuple[str, int]] = []  # terminal name and the number of arguments it takes
        for left_side, right_side in split_actions:
            right_side_is_number = False
            try:
                float(right_side)
                right_side_is_number = True
            except ValueError:
                pass
            if '[' in right_side:
                if 'lambda ' in right_side:
                    # "lambda x/y/...". Removing single quotes surrounding the string.
                    terminals.append((right_side[1:-1].split(', ')[0].replace("'", ""),
                                      self._infer_num_arguments(left_side)))
            elif len(right_side) > 1 or right_side in ['x', 'y', 'z', '<', '-'] or right_side_is_number:
                if right_side == "var" and add_var_function:
                    raise RuntimeError("You wanted to add var, but you didn't need to!")
                if right_side in ['x', 'y', 'z'] and add_var_function:
                    right_side = f"(var {right_side})"
                terminals.append((right_side, self._infer_num_arguments(left_side)))
        partial_logical_forms: List[str] = []

        # We'll handle "reverse" here by looking at the prior terminal to see if it's "reverse"
        # (and because we're iterating over this backwards, "prior" actually means "next").  If it
        # is, we'll create a new "terminal" that's the application of "reverse" to that terminal.
        # This works for most cases, but there are a few edge cases where this breaks, particularly
        # when there's a lambda involved.  We have a hack to try to handle lambdas, but it doesn't
        # always work.  TODO(mattg,pradeep): to make this logic more general and remove the need to
        # special-case reverse, we need to fix some things in the type system.  In particular, we
        # should remove currying in our action sequences.  If each production accurately reflected
        # the number of arguments to a function, we could remove most of this logic.
        terminals = list(reversed(terminals))
        for i, (terminal, num_args) in enumerate(terminals):
            if i < len(terminals) - 1 and terminals[i + 1][0] == 'reverse':
                if 'lambda' in terminal:
                    terminal = f"({terminal} {partial_logical_forms.pop()})"
                    num_args -= 1
                terminal = f"(reverse {terminal})"
            if terminal == 'reverse':
                continue
            elif num_args == 0:
                partial_logical_forms.append(terminal)
            else:
                args = []
                if len(partial_logical_forms) < num_args:
                    logger.error("Not enough arguments for: %s", terminal)
                    logger.error("Partial logical forms were: %s", partial_logical_forms)
                    logger.error("Action sequence was: %s", action_sequence)
                    raise ParsingError("Can't produce logical form from action sequence")
                for _ in range(num_args):
                    args.append(partial_logical_forms.pop())
                partial_logical_forms.append("(%s %s)" % (terminal, " ".join(args)))
        if len(partial_logical_forms) != 1:
            logger.error("Incomplete action sequence (or parsing error): %s", action_sequence)
            logger.error("Partial logical forms were: %s", partial_logical_forms)
            raise ParsingError("Can't produce logical form from action sequence")
        return partial_logical_forms[0]

    @classmethod
    def _infer_num_arguments(cls, type_signature: str) -> int:
        """
        Takes a type signature and infers the number of arguments the corresponding function takes.
        Examples:
            e -> 0
            <r,e> -> 1
            <e,<e,t>> -> 2
            <b,<<b,#1>,<#1,b>>> -> 3
        """
        if not "<" in type_signature:
            return 0
        # We need to find the return type from the signature. We do that by removing the outer most
        # angular brackets and traversing the remaining substring till the angular brackets (if any)
        # balance. Once we hit a comma after the angular brackets are balanced, whatever is left
        # after it is the return type.
        type_signature = type_signature[1:-1]
        num_brackets = 0
        char_index = 0
        for char in type_signature:
            if char == '<':
                num_brackets += 1
            elif char == '>':
                num_brackets -= 1
            elif char == ',':
                if num_brackets == 0:
                    break
            char_index += 1
        return_type = type_signature[char_index+1:]
        return 1 + cls._infer_num_arguments(return_type)

    def _process_nested_expression(self, nested_expression) -> str:
        """
        ``nested_expression`` is the result of parsing a logical form in Lisp format.
        We process it recursively and return a string in the format that NLTK's ``LogicParser``
        would understand.
        """
        expression_is_list = isinstance(nested_expression, list)
        expression_size = len(nested_expression)
        if expression_is_list and expression_size == 1 and isinstance(nested_expression[0], list):
            return self._process_nested_expression(nested_expression[0])
        elements_are_leaves = [isinstance(element, str) for element in nested_expression]
        if all(elements_are_leaves):
            mapped_names = [self._map_name(name) for name in nested_expression]
        else:
            mapped_names = []
            for element, is_leaf in zip(nested_expression, elements_are_leaves):
                if is_leaf:
                    mapped_names.append(self._map_name(element))
                else:
                    mapped_names.append(self._process_nested_expression(element))
        if mapped_names[0] == "\\":
            # This means the predicate is lambda. NLTK wants the variable name to not be within parantheses.
            # Adding parentheses after the variable.
            arguments = [mapped_names[1]] + ["(%s)" % name for name in mapped_names[2:]]
        else:
            arguments = ["(%s)" % name for name in mapped_names[1:]]
        return "(%s %s)" % (mapped_names[0], " ".join(arguments))

    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        """
        Takes the name of a predicate or a constant as used by Sempre, maps it to a unique string
        such that NLTK processes it appropriately. This is needed because NLTK has a naming
        convention for variables:

            - Function variables: Single upper case letter optionally followed by digits
            - Individual variables: Single lower case letter (except e for events) optionally
              followed by digits
            - Constants: Everything else

        Parameters
        ----------
        name : ``str``
            Token from Sempre's logical form.
        keep_mapping : ``bool``, optional (default=False)
            If this is ``True``, we will add the name and its mapping to our local state, so that
            :func:`get_name_mapping` and :func:`get_valid_actions` know about it.  You typically
            want to do this when you're `initializing` the object, but you very likely don't want
            to when you're parsing logical forms - getting an ill-formed logical form can then
            change your state in bad ways, for instance.
        """
        raise NotImplementedError

    def _add_name_mapping(self, name: str, translated_name: str, name_type: Type = None):
        """
        Utility method to add a name and its translation to the local name mapping, and the corresponding
        signature, if available to the local type signatures. This method also updates the reverse name
        mapping.
        """
        self.local_name_mapping[name] = translated_name
        self.reverse_name_mapping[translated_name] = name
        if name_type:
            self.local_type_signatures[translated_name] = name_type

    def _get_transitions(self,
                         expression: Expression,
                         current_transitions: List[str]) -> List[str]:
        expression_type = expression.type
        try:
            # ``Expression.visit()`` takes two arguments: the first one is a function applied on each
            # sub-expression and the second is a combinator that is applied to the list of values returned
            # from the function applications. We just want the list of all sub-expressions here.
            sub_expressions = expression.visit(lambda x: x, lambda x: x)
            transformed_types = [sub_exp.type for sub_exp in sub_expressions]
            if isinstance(expression, LambdaExpression):
                # If the expression is a lambda expression, the list of sub expressions does not include
                # the "lambda x" term. We're adding it here so that we will see transitions like
                #   <e,d> -> [\x, d] instead of
                #   <e,d> -> [d]
                transformed_types = ["lambda x"] + transformed_types
            current_transitions.append("%s -> %s" % (expression_type,
                                                     str(transformed_types)))
            for sub_expression in sub_expressions:
                self._get_transitions(sub_expression, current_transitions)
        except NotImplementedError:
            # This means that the expression is a leaf. We simply make a transition from its type to itself.
            original_name = str(expression)
            if original_name in self.reverse_name_mapping:
                original_name = self.reverse_name_mapping[original_name]
            current_transitions.append("%s -> %s" % (expression_type, original_name))
        return current_transitions

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented
