"""
We store all the information related to a world (i.e. the context in which logical forms will be
executed) here.
"""
from typing import Callable, Dict, List, Set
import re

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse.worlds.world import ParsingError, World
from allennlp.semparse.type_declarations import ratecalculus_type_declaration as types
from allennlp.semparse.knowledge_graphs import QuestionKnowledgeGraph


class RateCalculusWorld(World):
    """
    World representation for the Rate Calculus domain.

    Parameters
    ----------
    knowledge_graph : ``QuestionKnowledgeGraph``
        Context associated with this world.
    """
    # When we're converting from logical forms to action sequences, this set tells us which
    # functions in the logical form are curried functions, and how many arguments the function
    # actually takes.  This is necessary because NLTK curries all multi-argument functions to a
    # series of one-argument function applications.  See `world._get_transitions` for more info.
    curried_functions = {
            types.CONJUNCTION_TYPE: 2,
            types.BINARY_NUM_OP_TYPE: 2,
            types.BINARY_BOOL_OP_TYPE: 2,
            types.BINARY_NUM_TO_BOOL_OP_TYPE: 2,
            types.RATE_TYPE: 3,
            types.VALUE_TYPE: 2
            }

    def __init__(self, question_knowledge_graph: QuestionKnowledgeGraph) -> None:
        super(RateCalculusWorld, self).__init__(constant_type_prefixes={"num": types.NUMBER_TYPE},
                                                global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                              global_name_mapping=types.COMMON_NAME_MAPPING,
                                              num_nested_lambdas=0)
        self.question_knowledge_graph = question_knowledge_graph

        for entity in question_knowledge_graph.entities:
            self._map_name(entity, keep_mapping=True)

        self._entity_set = set(question_knowledge_graph.entities)

    def _get_curried_functions(self) -> Dict[Type, int]:
        return RateCalculusWorld.curried_functions

    def is_question_entity(self, entity_name: str) -> bool:
        """
        Returns ``True`` if the given entity is one of the entities in the question.
        """
        return entity_name in self._entity_set

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def get_valid_actions(self) -> Dict[str, List[str]]:
        valid_actions = super().get_valid_actions()

        return valid_actions


    @overrides
    def get_valid_starting_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        if name not in types.COMMON_NAME_MAPPING and name not in self.local_name_mapping:
            if not keep_mapping:
                raise ParsingError(f"Encountered un-mapped name: {name}")

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
