import re
from typing import List, Dict, Set
from collections import defaultdict
from sys import exc_info
from six import reraise
from overrides import overrides

from parsimonious.expressions import Literal, OneOf, Sequence
from parsimonious.nodes import Node, NodeVisitor
from parsimonious.grammar import Grammar
from parsimonious.exceptions import VisitationError, UndefinedLabel


WHITESPACE_REGEX = re.compile(" wsp |wsp | wsp| ws |ws | ws")

def format_grammar_string(grammar_dictionary: Dict[str, List[str]]) -> str:
    """
    Formats a dictionary of production rules into the string format expected
    by the Parsimonious Grammar class.
    """
    grammar_string = '\n'.join([f"{nonterminal} = {' / '.join(right_hand_side)}"
                                for nonterminal, right_hand_side in grammar_dictionary.items()])
    return grammar_string.replace("\\", "\\\\")


def initialize_valid_actions(grammar: Grammar,
                             keywords_to_uppercase: List[str] = None) -> Dict[str, List[str]]:
    """
    We initialize the valid actions with the global actions. These include the
    valid actions that result from the grammar and also those that result from
    the tables provided. The keys represent the nonterminals in the grammar
    and the values are lists of the valid actions of that nonterminal.
    """
    valid_actions: Dict[str, Set[str]] = defaultdict(set)

    for key in grammar:
        rhs = grammar[key]

        # Sequence represents a series of expressions that match pieces of the text in order.
        # Eg. A -> B C
        if isinstance(rhs, Sequence):
            valid_actions[key].add(format_action(key, " ".join(rhs._unicode_members()), # pylint: disable=protected-access
                                                 keywords_to_uppercase=keywords_to_uppercase))

        # OneOf represents a series of expressions, one of which matches the text.
        # Eg. A -> B / C
        elif isinstance(rhs, OneOf):
            for option in rhs._unicode_members(): # pylint: disable=protected-access
                valid_actions[key].add(format_action(key, option,
                                                     keywords_to_uppercase=keywords_to_uppercase))

        # A string literal, eg. "A"
        elif isinstance(rhs, Literal):
            if rhs.literal != "":
                valid_actions[key].add(format_action(key, repr(rhs.literal),
                                                     keywords_to_uppercase=keywords_to_uppercase))
            else:
                valid_actions[key] = set()

    valid_action_strings = {key: sorted(value) for key, value in valid_actions.items()}
    return valid_action_strings


def format_action(nonterminal: str,
                  right_hand_side: str,
                  is_string: bool = False,
                  is_number: bool = False,
                  keywords_to_uppercase: List[str] = None) -> str:
    """
    This function formats an action as it appears in models. It
    splits productions based on the special `ws` and `wsp` rules,
    which are used in grammars to denote whitespace, and then
    rejoins these tokens a formatted, comma separated list.
    Importantly, note that it `does not` split on spaces in
    the grammar string, because these might not correspond
    to spaces in the language the grammar recognises.

    Parameters
    ----------
    nonterminal : ``str``, required.
        The nonterminal in the action.
    right_hand_side : ``str``, required.
        The right hand side of the action
        (i.e the thing which is produced).
    is_string : ``bool``, optional (default = False).
        Whether the production produces a string.
        If it does, it is formatted as ``nonterminal -> ['string']``
    is_number : ``bool``, optional, (default = False).
        Whether the production produces a string.
        If it does, it is formatted as ``nonterminal -> ['number']``
    keywords_to_uppercase: ``List[str]``, optional, (default = None)
        Keywords in the grammar to uppercase. In the case of sql,
        this might be SELECT, MAX etc.
    """
    keywords_to_uppercase = keywords_to_uppercase or []
    if right_hand_side.upper() in keywords_to_uppercase:
        right_hand_side = right_hand_side.upper()

    if is_string:
        return f'{nonterminal} -> ["\'{right_hand_side}\'"]'

    elif is_number:
        return f'{nonterminal} -> ["{right_hand_side}"]'

    else:
        right_hand_side = right_hand_side.lstrip("(").rstrip(")")
        child_strings = [token for token in WHITESPACE_REGEX.split(right_hand_side) if token]
        child_strings = [tok.upper() if tok.upper() in keywords_to_uppercase else tok for tok in child_strings]
        return f"{nonterminal} -> [{', '.join(child_strings)}]"

def action_sequence_to_sql(action_sequences: List[str]) -> str:
    # Convert an action sequence like ['statement -> [query, ";"]', ...] to the
    # SQL string.
    query = []
    for action in action_sequences:
        nonterminal, right_hand_side = action.split(' -> ')
        right_hand_side_tokens = right_hand_side[1:-1].split(', ')
        if nonterminal == 'statement':
            query.extend(right_hand_side_tokens)
        else:
            for query_index, token in list(enumerate(query)):
                if token == nonterminal:
                    query = query[:query_index] + \
                            right_hand_side_tokens + \
                            query[query_index + 1:]
                    break
    return ' '.join([token.strip('"') for token in query])


class SqlVisitor(NodeVisitor):
    """
    ``SqlVisitor`` performs a depth-first traversal of the the AST. It takes the parse tree
    and gives us an action sequence that resulted in that parse. Since the visitor has mutable
    state, we define a new ``SqlVisitor`` for each query. To get the action sequence, we create
    a ``SqlVisitor`` and call parse on it, which returns a list of actions. Ex.

        sql_visitor = SqlVisitor(grammar_string)
        action_sequence = sql_visitor.parse(query)

    Importantly, this ``SqlVisitor`` skips over ``ws`` and ``wsp`` nodes,
    because they do not hold any meaning, and make an action sequence
    much longer than it needs to be.

    Parameters
    ----------
    grammar : ``Grammar``
        A Grammar object that we use to parse the text.
    keywords_to_uppercase: ``List[str]``, optional, (default = None)
        Keywords in the grammar to uppercase. In the case of sql,
        this might be SELECT, MAX etc.
    """
    def __init__(self, grammar: Grammar, keywords_to_uppercase: List[str] = None) -> None:
        self.action_sequence: List[str] = []
        self.grammar: Grammar = grammar
        self.keywords_to_uppercase = keywords_to_uppercase or []

    @overrides
    def generic_visit(self, node: Node, visited_children: List[None]) -> List[str]:
        self.add_action(node)
        if node.expr.name == 'statement':
            return self.action_sequence
        return []

    def add_action(self, node: Node) -> None:
        """
        For each node, we accumulate the rules that generated its children in a list.
        """
        if node.expr.name and node.expr.name not in ['ws', 'wsp']:
            nonterminal = f'{node.expr.name} -> '

            if isinstance(node.expr, Literal):
                right_hand_side = f'["{node.text}"]'

            else:
                child_strings = []
                for child in node.__iter__():
                    if child.expr.name in ['ws', 'wsp']:
                        continue
                    if child.expr.name != '':
                        child_strings.append(child.expr.name)
                    else:
                        child_right_side_string = child.expr._as_rhs().lstrip("(").rstrip(")") # pylint: disable=protected-access
                        child_right_side_list = [tok for tok in
                                                 WHITESPACE_REGEX.split(child_right_side_string) if tok]
                        child_right_side_list = [tok.upper() if tok.upper() in
                                                 self.keywords_to_uppercase else tok
                                                 for tok in child_right_side_list]
                        child_strings.extend(child_right_side_list)
                right_hand_side = "[" + ", ".join(child_strings) + "]"
            rule = nonterminal + right_hand_side
            self.action_sequence = [rule] + self.action_sequence

    @overrides
    def visit(self, node):
        """
        See the ``NodeVisitor`` visit method. This just changes the order in which
        we visit nonterminals from right to left to left to right.
        """
        method = getattr(self, 'visit_' + node.expr_name, self.generic_visit)

        # Call that method, and show where in the tree it failed if it blows
        # up.
        try:
            # Changing this to reverse here!
            return method(node, [self.visit(child) for child in reversed(list(node))])
        except (VisitationError, UndefinedLabel):
            # Don't catch and re-wrap already-wrapped exceptions.
            raise
        except self.unwrapped_exceptions:
            raise
        except Exception: # pylint: disable=broad-except
            # Catch any exception, and tack on a parse tree so it's easier to
            # see where it went wrong.
            exc_class, exc, traceback = exc_info()
            reraise(VisitationError, VisitationError(exc, exc_class, node), traceback)
