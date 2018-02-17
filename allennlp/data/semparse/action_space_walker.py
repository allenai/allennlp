from collections import defaultdict
from typing import List, Dict, Set

from allennlp.data.semparse.type_declarations import type_declaration as types
from allennlp.data.semparse.worlds.world import World


class ActionSpaceWalker:
    """
    ``ActionSpaceWalker`` takes a world, traverses all the valid paths driven by the valid action
    specification of the world to generate all possible logical forms (under some constraints). This
    class also has some utilities for indexing logical forms to efficiently retrieve required
    subsets.

    Parameters
    ----------
    """
    def __init__(self, world: World) -> None:
        self._world = world
        self._completed_paths: List[List[str]] = None
        self._terminal_path_index: Dict[str, Set[int]] = defaultdict(set)

    def walk(self, max_path_length: int) -> None:
        """
        Walk over action space to collect completed paths of at most ``max_path_length`` steps.
        """
        # Buffer of NTs to expand, previous actions
        incomplete_paths = [([str(type_)], [f"@START@ -> {type_}"]) for type_ in
                            self._world.get_valid_starting_types()]

        self._completed_paths = []
        actions = self._world.get_valid_actions()
        while incomplete_paths:
            new_incomplete_paths = []
            for nonterminal_buffer, history in incomplete_paths:
                nonterminal = nonterminal_buffer.pop()
                for action in actions[nonterminal]:
                    new_history = history + [action]
                    new_nonterminal_buffer = list(nonterminal_buffer)
                    for rhs_part in reversed(self._get_right_side_parts(action)):
                        if types.is_nonterminal(rhs_part):
                            new_nonterminal_buffer.append(rhs_part)
                    new_incomplete_paths.append((new_nonterminal_buffer, new_history))
            incomplete_paths = []
            for nonterminal_buffer, path in new_incomplete_paths:
                if not nonterminal_buffer:
                    # Indexing completed paths by the nonterminals they contain.
                    next_path_index = len(self._completed_paths)
                    for action in path:
                        for value in self._get_right_side_parts(action):
                            if not types.is_nonterminal(value):
                                self._terminal_path_index[action].add(next_path_index)
                    self._completed_paths.append(path)
                elif len(path) <= max_path_length:
                    incomplete_paths.append((nonterminal_buffer, path))

    @staticmethod
    def _get_right_side_parts(action: str) -> List[str]:
        _, right_side = action.split(" -> ")
        right_side_parts = []
        if "[" in right_side:
            right_side_parts = right_side[1:-1].split(", ")
        else:
            right_side_parts.append(right_side)
        return right_side_parts

    def get_logical_forms_with_agenda(self, agenda: List[str]) -> List[str]:
        if self._completed_paths is None:
            raise RuntimeError("Run ActionSpaceWalker.walk() first!")
        agenda_path_indices = [self._terminal_path_index[action] for action in agenda]
        # TODO (pradeep): Sort the indices and do intersections in order, so that we can return the
        # set with maximal coverage if the full intersection is null.
        return_set = agenda_path_indices[0]
        for next_set in agenda_path_indices[1:]:
            return_set = return_set.intersection(next_set)
        paths = [self._completed_paths[index] for index in return_set]
        logical_forms = [self._world.get_logical_form(path) for path in paths]
        return logical_forms
