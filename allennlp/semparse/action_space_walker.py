from collections import defaultdict
from typing import List, Dict, Set
import logging

from allennlp.common.util import START_SYMBOL
from allennlp.semparse.domain_languages.domain_language import DomainLanguage


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ActionSpaceWalker:
    """
    ``ActionSpaceWalker`` takes a world, traverses all the valid paths driven by the valid action
    specification of the world to generate all possible logical forms (under some constraints). This
    class also has some utilities for indexing logical forms to efficiently retrieve required
    subsets.

    Parameters
    ----------
    world : ``DomainLanguage``
        The world (domain language instantiation) from which valid actions will be taken.
    max_path_length : ``int``
        The maximum path length till which the action space will be explored. Paths longer than this
        length will be discarded.
    """
    def __init__(self, world: DomainLanguage, max_path_length: int) -> None:
        self._world = world
        self._max_path_length = max_path_length
        self._completed_paths: List[List[str]] = None
        self._terminal_path_index: Dict[str, Set[int]] = defaultdict(set)
        self._length_sorted_paths: List[List[str]] = None

    def _walk(self) -> None:
        """
        Walk over action space to collect completed paths of at most ``self._max_path_length`` steps.
        """
        actions = self._world.get_nonterminal_productions()
        start_productions = actions[START_SYMBOL]
        # Buffer of NTs to expand, previous actions
        incomplete_paths = [([start_production.split(' -> ')[-1]], [start_production])
                            for start_production in start_productions]
        self._completed_paths = []
        # Overview: We keep track of the buffer of non-terminals to expand, and the action history
        # for each incomplete path. At every iteration in the while loop below, we iterate over all
        # incomplete paths, expand one non-terminal from the buffer in a depth-first fashion, get
        # all possible next actions triggered by that non-terminal and add to the paths. Then, we
        # check the expanded paths, to see if they are 1) complete, in which case they are
        # added to completed_paths, 2) longer than max_path_length, in which case they are
        # discarded, or 3) neither, in which case they are used to form the incomplete_paths for the
        # next iteration of this while loop.
        # While the non-terminal expansion is done in a depth-first fashion, note that the search over
        # the action space itself is breadth-first.
        while incomplete_paths:
            next_paths = []
            for nonterminal_buffer, history in incomplete_paths:
                # Taking the last non-terminal added to the buffer. We're going depth-first.
                nonterminal = nonterminal_buffer.pop()
                next_actions = []
                if nonterminal not in actions:
                    # This happens when the nonterminal corresponds to a type that does not exist in
                    # the context. For example, in the variable free variant of the WikiTables
                    # world, there are nonterminals for specific column types (like date). Say we
                    # produced a path containing "filter_date_greater" already, and we do not have
                    # an columns of type "date", then this condition would be triggered. We should
                    # just discard those paths.
                    continue
                else:
                    next_actions.extend(actions[nonterminal])
                # Iterating over all possible next actions.
                for action in next_actions:
                    new_history = history + [action]
                    new_nonterminal_buffer = nonterminal_buffer[:]
                    # Since we expand the last action added to the buffer, the left child should be
                    # added after the right child.
                    for right_side_part in reversed(self._get_right_side_parts(action)):
                        if self._world.is_nonterminal(right_side_part):
                            new_nonterminal_buffer.append(right_side_part)
                    next_paths.append((new_nonterminal_buffer, new_history))
            incomplete_paths = []
            for nonterminal_buffer, path in next_paths:
                # An empty buffer means that we've completed this path.
                if not nonterminal_buffer:
                    # Indexing completed paths by the nonterminals they contain.
                    next_path_index = len(self._completed_paths)
                    for action in path:
                        for value in self._get_right_side_parts(action):
                            if not self._world.is_nonterminal(value):
                                self._terminal_path_index[action].add(next_path_index)
                    self._completed_paths.append(path)
                # We're adding to incomplete_paths for the next iteration, only those paths that are
                # shorter than the max_path_length. The remaining paths will be discarded.
                elif len(path) <= self._max_path_length:
                    incomplete_paths.append((nonterminal_buffer, path))

    @staticmethod
    def _get_right_side_parts(action: str) -> List[str]:
        _, right_side = action.split(" -> ")
        if right_side.startswith("["):
            right_side_parts = right_side[1:-1].split(", ")
        else:
            right_side_parts = [right_side]
        return right_side_parts

    def get_logical_forms_with_agenda(self,
                                      agenda: List[str],
                                      max_num_logical_forms: int = None,
                                      allow_partial_match: bool = False) -> List[str]:
        """
        Parameters
        ----------
        agenda : ``List[str]``
        max_num_logical_forms : ``int`` (optional)
        allow_partial_match : ``bool`` (optional, defaul=False)
            If set, this method will return logical forms which contain not necessarily all the
            items on the agenda. The returned list will be sorted by how many items the logical
            forms match.
        """
        if not agenda:
            if allow_partial_match:
                logger.warning("Agenda is empty! Returning all paths instead.")
                return self.get_all_logical_forms(max_num_logical_forms)
            return []
        if self._completed_paths is None:
            self._walk()
        agenda_path_indices = [self._terminal_path_index[action] for action in agenda]
        if all([not path_indices for path_indices in agenda_path_indices]):
            if allow_partial_match:
                logger.warning("""Agenda items not in any of the paths found. Returning all paths.""")
                return self.get_all_logical_forms(max_num_logical_forms)
            return []
        # TODO (pradeep): Sort the indices and do intersections in order, so that we can return the
        # set with maximal coverage if the full intersection is null.

        # This list contains for each agenda item the list of indices of paths that contain that agenda item. Note
        # that we omit agenda items that are not in any paths to avoid the final intersection being null. So there
        # will not be any empty sub-lists in the list below.
        filtered_path_indices: List[Set[int]] = []
        for agenda_item, path_indices in zip(agenda, agenda_path_indices):
            if not path_indices:
                logger.warning(f"{agenda_item} is not in any of the paths found! Ignoring it.")
                continue
            filtered_path_indices.append(path_indices)

        # This mapping is from a path index to the number of items in the agenda that the path contains.
        index_to_num_items: Dict[int, int] = defaultdict(int)
        for indices in filtered_path_indices:
            for index in indices:
                index_to_num_items[index] += 1
        if allow_partial_match:
            # We group the paths based on how many agenda items they contain, and output them in a sorted order.
            num_items_grouped_paths: Dict[int, List[List[str]]] = defaultdict(list)
            for index, num_items in index_to_num_items.items():
                num_items_grouped_paths[num_items].append(self._completed_paths[index])
            paths = []
            # Sort by number of agenda items present in the paths.
            for num_items, corresponding_paths in sorted(num_items_grouped_paths.items(),
                                                         reverse=True):
                # Given those paths, sort them by length, so that the first path in ``paths`` will
                # be the shortest path with the most agenda items.
                paths.extend(sorted(corresponding_paths, key=len))
        else:
            indices_to_return = []
            for index, num_items in index_to_num_items.items():
                if num_items == len(filtered_path_indices):
                    indices_to_return.append(index)
            # Sort all the paths by length
            paths = sorted([self._completed_paths[index] for index in indices_to_return], key=len)
        if max_num_logical_forms is not None:
            paths = paths[:max_num_logical_forms]
        logical_forms = [self._world.action_sequence_to_logical_form(path) for path in paths]
        return logical_forms

    def get_all_logical_forms(self,
                              max_num_logical_forms: int = None) -> List[str]:
        if self._completed_paths is None:
            self._walk()
        paths = self._completed_paths
        if max_num_logical_forms is not None:
            if self._length_sorted_paths is None:
                self._length_sorted_paths = sorted(self._completed_paths, key=len)
            paths = self._length_sorted_paths[:max_num_logical_forms]
        logical_forms = [self._world.action_sequence_to_logical_form(path) for path in paths]
        return logical_forms
