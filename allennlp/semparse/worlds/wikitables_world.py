"""
We store all the information related to a world (i.e. the context in which logical forms will be
executed) here. For WikiTableQuestions, this includes a representation of a table, mapping from
Sempre variables in all logical forms to NLTK variables, and the types of all predicates and entities.
"""
from typing import Callable, Dict, List, Set
import re

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse.worlds.world import ParsingError, World
from allennlp.semparse.type_declarations import wikitables_type_declaration as types
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph


class WikiTablesWorld(World):
    """
    World representation for the WikitableQuestions domain.

    Parameters
    ----------
    table_graph : ``TableQuestionKnowledgeGraph``
        Context associated with this world.
    """
    # When we're converting from logical forms to action sequences, this set tells us which
    # functions in the logical form are curried functions, and how many arguments the function
    # actually takes.  This is necessary because NLTK curries all multi-argument functions to a
    # series of one-argument function applications.  See `world._get_transitions` for more info.
    curried_functions = {
            types.ARG_EXTREME_TYPE: 4,
            types.CONJUNCTION_TYPE: 2,
            types.DATE_FUNCTION_TYPE: 3,
            types.BINARY_NUM_OP_TYPE: 2,
            }

    def __init__(self, table_graph: TableQuestionKnowledgeGraph) -> None:
        super(WikiTablesWorld, self).__init__(constant_type_prefixes={"part": types.PART_TYPE,
                                                                      "cell": types.CELL_TYPE,
                                                                      "num": types.NUMBER_TYPE},
                                              global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                              global_name_mapping=types.COMMON_NAME_MAPPING,
                                              num_nested_lambdas=1)
        self.table_graph = table_graph

        # For every new Sempre column name seen, we update this counter to map it to a new NLTK name.
        self._column_counter = 0

        # This adds all of the cell and column names to our local name mapping, including null
        # cells and columns and a few simple numbers, so we can get them as valid actions in the
        # parser.  The null cell and column are used to check against empty sets, e.g., for
        # questions like "Is there a team that won three times in a row?".
        for entity in table_graph.entities + ['fb:cell.null', 'fb:row.row.null', '-1', '0', '1']:
            self._map_name(entity, keep_mapping=True)

        self._entity_set = set(table_graph.entities)
        self.terminal_productions: Dict[str, str] = {}
        for entity in self._entity_set:
            mapped_name = self.local_name_mapping[entity]
            signature = self.local_type_signatures[mapped_name]
            self.terminal_productions[entity] = f"{signature} -> {entity}"

        for predicate, mapped_name in self.global_name_mapping.items():
            if mapped_name in self.global_type_signatures:
                signature = self.global_type_signatures[mapped_name]
                self.terminal_productions[predicate] = f"{signature} -> {predicate}"

    def is_table_entity(self, entity_name: str) -> bool:
        """
        Returns ``True`` if the given entity is one of the entities in the table.
        """
        return entity_name in self._entity_set

    def _get_curried_functions(self) -> Dict[Type, int]:
        return WikiTablesWorld.curried_functions

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def get_valid_actions(self) -> Dict[str, List[str]]:
        valid_actions = super().get_valid_actions()

        # We need to add a few things here that don't get added by our world-general logic, and
        # remove some things that are technically possible in our type system, but not present in
        # the original SEMPRE grammar.

        # These are possible because of `reverse`.
        valid_actions['c'].append('c -> [<r,c>, r]')
        valid_actions['d'].append('d -> [<c,d>, c]')
        valid_actions['n'].append('n -> [<r,n>, r]')
        valid_actions['n'].append('n -> [<c,n>, c]')
        valid_actions['p'].append('p -> [<c,p>, c]')
        valid_actions['<p,n>'] = ["<p,n> -> ['lambda x', n]"]

        # These get added when we do our ANY_TYPE substitution with basic types, but we don't
        # actually need them.
        del valid_actions['<c,c>']
        del valid_actions['<d,p>']
        del valid_actions['<p,d>']
        del valid_actions['<p,p>']
        del valid_actions['<p,r>']

        # The argmax type generates an action that takes a date as an argument, which it turns out
        # we don't need for parts.
        self._remove_action_from_type(valid_actions, 'p', lambda x: '<d,p>' in x)

        # Our code that generates lambda productions similarly creates more than we need.
        for type_ in ['<c,p>', '<c,r>', '<d,c>', '<d,r>', '<n,c>', '<n,p>', '<n,r>', '<p,c>',
                      '<r,c>', '<r,r>']:
            self._remove_action_from_type(valid_actions, type_, lambda x: 'lambda' in x)

        # And we don't need `reverse` productions everywhere they are added, either.
        for type_ in ['<c,r>', '<p,c>', '<r,d>']:
            self._remove_action_from_type(valid_actions, type_, lambda x: '<<#1,#2>,<#2,#1>>' in x)

        return valid_actions

    @staticmethod
    def _remove_action_from_type(valid_actions: Dict[str, List[str]],
                                 type_: str,
                                 filter_function: Callable[[str], bool]) -> None:
        """
        Finds the production rule matching the filter function in the given type's valid action
        list, and removes it.  If there is more than one matching function, we crash.
        """
        action_list = valid_actions[type_]
        matching_action_index = [i for i, action in enumerate(action_list) if filter_function(action)]
        assert len(matching_action_index) == 1, "Filter function didn't find one action"
        action_list.pop(matching_action_index[0])

    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        if name not in types.COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if not keep_mapping:
                raise ParsingError(f"Encountered un-mapped name: {name}")
            if name.startswith("fb:row.row"):
                # Column name
                translated_name = "C%d" % self._column_counter
                self._column_counter += 1
                self._add_name_mapping(name, translated_name, types.COLUMN_TYPE)
            elif name.startswith("fb:cell"):
                # Cell name
                translated_name = "cell:%s" % name.split(".")[-1]
                self._add_name_mapping(name, translated_name, types.CELL_TYPE)
            elif name.startswith("fb:part"):
                # part name
                translated_name = "part:%s" % name.split(".")[-1]
                self._add_name_mapping(name, translated_name, types.PART_TYPE)
            else:
                # The only other unmapped names we should see are numbers.
                # NLTK throws an error if it sees a "." in constants, which will most likely happen
                # within numbers as a decimal point. We're changing those to underscores.
                translated_name = name.replace(".", "_")
                if re.match("-[0-9_]+", translated_name):
                    # The string is a negative number. This makes NLTK interpret this as a negated
                    # expression and force its type to be TRUTH_VALUE (t).
                    translated_name = translated_name.replace("-", "~")
                translated_name = f"num:{translated_name}"
                self._add_name_mapping(name, translated_name, types.NUMBER_TYPE)
        else:
            if name in types.COMMON_NAME_MAPPING:
                translated_name = types.COMMON_NAME_MAPPING[name]
            else:
                translated_name = self.local_name_mapping[name]
        return translated_name

    def get_agenda(self):
        agenda_items = self.table_graph.get_linked_agenda_items()
        # Global rules
        question_tokens = [token.text for token in self.table_graph.question_tokens]
        question = " ".join(question_tokens)
        for token in question_tokens:
            if token in ["next", "previous", "before", "after", "above", "below"]:
                agenda_items.append("fb:row.row.next")
            if token == "total":
                agenda_items.append("sum")
            if token == "difference":
                agenda_items.append("-")
            if token == "average":
                agenda_items.append("avg")
            if token in ["least", "top", "first", "smallest", "shortest", "lowest"]:
                # This condition is too brittle. But for most logical forms with "min", there are
                # semantically equivalent ones with "argmin". The exceptions are rare.
                if "what is the least" in question:
                    agenda_items.append("min")
                else:
                    agenda_items.append("argmin")
            if token in ["last", "most", "largest", "highest", "longest", "greatest"]:
                # This condition is too brittle. But for most logical forms with "max", there are
                # semantically equivalent ones with "argmax". The exceptions are rare.
                if "what is the most" in question:
                    agenda_items.append("max")
                else:
                    agenda_items.append("argmax")

        if "how many" in question or "number" in question:
            if "sum" not in agenda_items and "avg" not in agenda_items:
                # The question probably just requires counting the rows. But this is not very
                # accurate. The question could also be asking for a value that is in the table.
                agenda_items.append("count")
        agenda = []
        for agenda_item in set(agenda_items):
            agenda.append(self.terminal_productions[agenda_item])
        return agenda
