from typing import List, Dict, Set
import logging

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.semparse.domain_languages.nlvr_language import Box, NlvrLanguage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NlvrWorld:
    """
    Class defining the world representation of NLVR. Defines an execution logic for logical forms
    in NLVR.  We just take the structured_rep from the JSON file to initialize this.

    Parameters
    ----------
    world_representation : ``JsonDict``
        structured_rep from the JSON file.
    """

    # TODO(pradeep): Define more spatial relationship methods: left_of, right_of..
    # They should be defined for objects within the same box.
    def __init__(self, world_representation: List[List[JsonDict]]) -> None:
        boxes = set([Box(object_list, box_id) for box_id, object_list in
                     enumerate(world_representation)])
        self._language = NlvrLanguage(boxes)

        # Mapping from terminal strings to productions that produce them.
        # Eg.: "yellow" -> "<o,o> -> yellow", "<b,<<b,e>,<e,b>>> -> filter_greater" etc.
        self.terminal_productions: Dict[str, str] = {}
        for name, type_ in self._language._function_types.items():  # pylint: disable=protected-access
            self.terminal_productions[name] = "%s -> %s" % (type_, name)

    def get_valid_actions(self):
        return self._language.get_nonterminal_productions()

    def all_possible_actions(self):
        return self._language.all_possible_productions()

    def get_logical_form(self, action_sequence: List[str]) -> str:
        return self._language.action_sequence_to_logical_form(action_sequence)

    def get_agenda_for_sentence(self,
                                sentence: str,
                                add_paths_to_agenda: bool = False) -> List[str]:
        """
        Given a ``sentence``, returns a list of actions the sentence triggers as an ``agenda``. The
        ``agenda`` can be used while by a parser to guide the decoder.  sequences as possible. This
        is a simplistic mapping at this point, and can be expanded.

        Parameters
        ----------
        sentence : ``str``
            The sentence for which an agenda will be produced.
        add_paths_to_agenda : ``bool`` , optional
            If set, the agenda will also include nonterminal productions that lead to the terminals
            from the root node (default = False).
        """
        agenda = []
        sentence = sentence.lower()
        if sentence.startswith("there is a box") or sentence.startswith("there is a tower "):
            agenda.append(self.terminal_productions["box_exists"])
        elif sentence.startswith("there is a "):
            agenda.append(self.terminal_productions["object_exists"])

        if "<Set[Box]:bool> -> box_exists" not in agenda:
            # These are object filters and do not apply if we have a box_exists at the top.
            if "touch" in sentence:
                if "top" in sentence:
                    agenda.append(self.terminal_productions["touch_top"])
                elif "bottom" in sentence or "base" in sentence:
                    agenda.append(self.terminal_productions["touch_bottom"])
                elif "corner" in sentence:
                    agenda.append(self.terminal_productions["touch_corner"])
                elif "right" in sentence:
                    agenda.append(self.terminal_productions["touch_right"])
                elif "left" in sentence:
                    agenda.append(self.terminal_productions["touch_left"])
                elif "wall" in sentence or "edge" in sentence:
                    agenda.append(self.terminal_productions["touch_wall"])
                else:
                    agenda.append(self.terminal_productions["touch_object"])
            else:
                # The words "top" and "bottom" may be referring to top and bottom blocks in a tower.
                if "top" in sentence:
                    agenda.append(self.terminal_productions["top"])
                elif "bottom" in sentence or "base" in sentence:
                    agenda.append(self.terminal_productions["bottom"])

            if " not " in sentence:
                agenda.append(self.terminal_productions["negate_filter"])

        if " contains " in sentence or " has " in sentence:
            agenda.append(self.terminal_productions["all_boxes"])
        # This takes care of shapes, colors, top, bottom, big, small etc.
        for constant, production in self.terminal_productions.items():
            # TODO(pradeep): Deal with constant names with underscores.
            if "top" in constant or "bottom" in constant:
                # We already dealt with top, bottom, touch_top and touch_bottom above.
                continue
            if constant in sentence:
                if "<Set[Object]:Set[Object]> ->" in production and "<Set[Box]:bool> -> box_exists" in agenda:
                    if constant in ["square", "circle", "triangle"]:
                        agenda.append(self.terminal_productions[f"shape_{constant}"])
                    elif constant in ["yellow", "blue", "black"]:
                        agenda.append(self.terminal_productions[f"color_{constant}"])
                    else:
                        continue
                else:
                    agenda.append(production)
        # TODO (pradeep): Rules for "member_*" productions ("tower" or "box" followed by a color,
        # shape or number...)
        number_productions = self._get_number_productions(sentence)
        for production in number_productions:
            agenda.append(production)
        if not agenda:
            # None of the rules above was triggered!
            if "box" in sentence:
                agenda.append(self.terminal_productions["all_boxes"])
            else:
                agenda.append(self.terminal_productions["all_objects"])
        if add_paths_to_agenda:
            agenda = self._add_nonterminal_productions(agenda)
        return agenda

    @staticmethod
    def _get_number_productions(sentence: str) -> List[str]:
        """
        Gathers all the numbers in the sentence, and returns productions that lead to them.
        """
        # The mapping here is very simple and limited, which also shouldn't be a problem
        # because numbers seem to be represented fairly regularly.
        number_strings = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six":
                          "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        number_productions = []
        tokens = sentence.split()
        numbers = number_strings.values()
        for token in tokens:
            if token in numbers:
                number_productions.append(f"int -> {token}")
            elif token in number_strings:
                number_productions.append(f"int -> {number_strings[token]}")
        return number_productions

    def _add_nonterminal_productions(self, agenda: List[str]) -> List[str]:
        """
        Given a partially populated agenda with (mostly) terminal productions, this method adds the
        nonterminal productions that lead from the root to the terminal productions.
        """
        nonterminal_productions = set(agenda)
        for action in agenda:
            paths = self.get_paths_to_root(action, max_num_paths=5)
            for path in paths:
                for path_action in path:
                    nonterminal_productions.add(path_action)
        new_agenda = list(nonterminal_productions)
        return new_agenda

    def execute(self, logical_form: str) -> bool:
        return self._language.execute(logical_form)
