from typing import Callable, Dict, List, Set
import re

from nltk.sem.logic import Type
from overrides import overrides

from allennlp.semparse.worlds.world import ParsingError, World
from allennlp.semparse.type_declarations import atis_type_declaration as types

# Import contexts?

class AtisWorld(World):
    def __init__(self) -> None:
        super(AtisWorld, self).__init__(constant_type_prefixes={"string": types.STRING_TYPE,
                                                                "num": types.NUM_TYPE},
                                        global_type_signatures=types.COMMON_TYPE_SIGNATURE,
                                        global_name_mapping=types.COMMON_NAME_MAPPING)
    curried_functions = {
            types.CONJ_TYPE: 2,
            types.BINOP_TYPE: 2,
            types.SELECT_TYPE: 3,
            types.FROM_TYPE: 1,
            types.WHERE_TYPE: 1}

    def _get_curried_functions(self) -> Dict[Type, int]:
        return AtisWorld.curried_functions

    @overrides
    def get_basic_types(self) -> Set[Type]:
        return types.BASIC_TYPES

    @overrides
    def _map_name(self, name: str, keep_mapping: bool = False) -> str:
        return types.COMMON_NAME_MAPPING[name] if name in types.COMMON_NAME_MAPPING else name
