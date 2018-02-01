from typing import Dict, List
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, ListField, IndexField, LabelField
from allennlp.data.fields import ProductionRuleField, MetadataField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.semparse.type_declarations import type_declaration as types
from allennlp.data.semparse.type_declarations import nlvr_type_declaration as nlvr_types
from allennlp.data.semparse.worlds import NlvrWorld


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("nlvr")
class NlvrDatasetReader(DatasetReader):
    """
    ``DatasetReader`` for the NLVR domain. In addition to the usual methods for reading files and
    instances from text, this class contains a method for creating an agenda of actions that each
    sentence triggers.

    We process the json version of the dataset (http://lic.nlp.cornell.edu/nlvr/) here, to read in
    the structured representations of the synthetic images instead of dealing with the actual images
    themselves. The format of each line in the jsonl file is
    ```
    "sentence": <sentence>,
    "label": <true/false>,
    "identifier": <id>,
    "evals": <dict containing all annotations>,
    "structured_rep": <list of three box representations, where each box is a list of object
    representation dicts, containing fields "x_loc", "y_loc", "color", "type", "size">
    ```
    We use the fields ``sentence``, ``label`` and ``structured_rep``.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    tokenizer : ``Tokenizer`` (optional)
        The tokenizer used for sentences in NLVR. Default is ``WordTokenizer``
    sentence_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for tokens in input sentences.
        Default is ``{"tokens": SingleIdTokenIndexer()}``
    nonterminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for non-terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    terminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    add_paths_to_agenda : ``bool`` (optional)
        If set to true, the agenda will also contain the non-terminal productions in the path from
        the root node to each of the desired terminal productions. We do an approximate heuristic
        search while computing the paths to avoid infinitely long ones (containing cycles).
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 sentence_token_indexers: Dict[str, TokenIndexer] = None,
                 nonterminal_indexers: Dict[str, TokenIndexer] = None,
                 terminal_indexers: Dict[str, TokenIndexer] = None,
                 add_paths_to_agenda: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_token_indexers = sentence_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._nonterminal_indexers = nonterminal_indexers or {"tokens":
                                                              SingleIdTokenIndexer("rule_labels")}
        self._terminal_indexers = terminal_indexers or {"tokens": SingleIdTokenIndexer("rule_labels")}
        # Mapping from terminal strings to productions that produce them.
        # Eg.: "yellow" -> "<o,o> -> yellow", "<b,<<b,e>,<e,b>>> -> filter_greater" etc.
        self._terminal_productions: Dict[str, str] = {}
        self._add_paths_to_agenda = add_paths_to_agenda
        for constant in nlvr_types.COMMON_NAME_MAPPING:
            alias = nlvr_types.COMMON_NAME_MAPPING[constant]
            if alias in nlvr_types.COMMON_TYPE_SIGNATURE:
                constant_type = nlvr_types.COMMON_TYPE_SIGNATURE[alias]
                if constant_type != types.ANY_TYPE:
                    self._terminal_productions[constant] = "%s -> %s" % (constant_type, constant)

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                data = json.loads(line)
                sentence = data["sentence"]
                label = data["label"]
                structured_representation = data["structured_rep"]
                yield self.text_to_instance(sentence, structured_representation, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str,
                         structured_representation: List[List[JsonDict]],
                         label: str = None) -> Instance:
        """
        Parameters
        ----------
        sentence : ``str``
            The query sentence.
        structured_representation : ``List[List[JsonDict]]``
            A Json representation of the context. See expected format in this class' docstring.
        label : ``str`` (optional)
            String representation of the label (true or false). Not required while testing.
        """
        # pylint: disable=arguments-differ
        world = NlvrWorld(structured_representation)
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)
        agenda = self._get_agenda_for_sentence(sentence, world)
        assert agenda, "No agenda found for sentence: %s" % sentence
        production_rule_fields: List[Field] = []
        instance_action_ids: Dict[str, int] = {}
        for production_rule in world.all_possible_actions():
            instance_action_ids[production_rule] = len(instance_action_ids)
            field = ProductionRuleField(production_rule,
                                        terminal_indexers=self._terminal_indexers,
                                        nonterminal_indexers=self._nonterminal_indexers,
                                        is_nonterminal=lambda x: x not in self._terminal_productions,
                                        context=tokenized_sentence)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)
        # agenda_field contains indices into actions.
        agenda_field = ListField([IndexField(instance_action_ids[action], action_field)
                                  for action in agenda])
        world_field = MetadataField(world)
        fields = {"sentence": sentence_field,
                  "agenda": agenda_field,
                  "world": world_field,
                  "actions": action_field}
        if label:
            label_field = LabelField(label, label_namespace='denotations')
            fields["label"] = label_field
        return Instance(fields)

    def _get_agenda_for_sentence(self, sentence: str, world: NlvrWorld = None) -> List[str]:
        """
        Given a ``sentence``, and a corresponding ``NlvrWorld`` returns a list of actions the
        sentence triggers. The model tries to include as many of these actions in the decoded
        sequences as possible. Hence, this method defines a mapping from sentence level features to
        actions. This is a simplistic mapping at this point, and can be expanded.
        """
        # TODO(pradeep): Add more rules in the mapping?
        # TODO(pradeep): Use approximate and substring matching as well.
        agenda = []
        # This takes care of shapes, colors, top, bottom, big, small etc.
        for constant, production in self._terminal_productions.items():
            # TODO(pradeep): Deal with constant names with underscores.
            if constant in sentence:
                agenda.append(production)
        if sentence.startswith("There is a "):
            agenda.append(self._terminal_productions["assert_greater_equals"])
        if "tower" in sentence or "box" in sentence or "grey" in sentence:
            # Ensuring box filtering function (filter_*) at top.
            agenda.append("t -> [<b,t>, b]")
        if "touch" in sentence:
            if "top" in sentence:
                agenda.append(self._terminal_productions["touch_top"])
            elif "bottom" in sentence:
                agenda.append(self._terminal_productions["touch_bottom"])
            elif "corner" in sentence:
                agenda.append(self._terminal_productions["touch_corner"])
            elif "wall" or "edge" in sentence:
                agenda.append(self._terminal_productions["touch_wall"])
        else:
            # The words "top" and "bottom" may be referring to top and bottom blocks in a tower.
            if "top" in sentence:
                agenda.append(self._terminal_productions["top"])
            elif "bottom" in sentence or "base" in sentence:
                agenda.append(self._terminal_productions["bottom"])
        if " not " in sentence:
            agenda.append(self._terminal_productions["negate_filter"])
        number_productions = self._get_number_productions(sentence)
        if number_productions or set():
            agenda.append(self._terminal_productions["count"])
        for production in number_productions:
            agenda.append(production)
        if self._add_paths_to_agenda:
            assert world is not None, "Pass a world if you want to add paths to the agenda"
            agenda = self._add_nonterminal_productions(agenda, world)
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
        for word, numeral in number_strings.items():
            if word in sentence or numeral in sentence:
                number_productions.append(f"e -> {numeral}")
        if "there is a " in sentence:
            number_productions.append("e -> 1")
        return number_productions

    @staticmethod
    def _add_nonterminal_productions(agenda: List[str], world: NlvrWorld) -> List[str]:
        """
        Given a partially populated agenda with (mostly) terminal productions, this method adds the
        nonterminal productions that lead from the root to the terminal productions.
        """
        nonterminal_productions = set(agenda)
        for action in agenda:
            paths = world.get_paths_to_root(action, max_num_paths=5)
            for path in paths:
                for path_action in path:
                    nonterminal_productions.add(path_action)
        new_agenda = list(nonterminal_productions)
        return new_agenda

    @classmethod
    def from_params(cls, params: Params) -> 'NlvrDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        sentence_token_indexers = TokenIndexer.dict_from_params(params.pop('sentence_token_indexers', {}))
        terminal_indexers = TokenIndexer.dict_from_params(params.pop('terminal_indexers', {}))
        nonterminal_indexers = TokenIndexer.dict_from_params(params.pop('nonterminal_indexers', {}))
        add_paths_to_agenda = params.pop("add_paths_to_agenda", True)
        params.assert_empty(cls.__name__)
        return NlvrDatasetReader(lazy=lazy,
                                 tokenizer=tokenizer,
                                 sentence_token_indexers=sentence_token_indexers,
                                 terminal_indexers=terminal_indexers,
                                 nonterminal_indexers=nonterminal_indexers,
                                 add_paths_to_agenda=add_paths_to_agenda)
