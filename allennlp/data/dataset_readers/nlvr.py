from typing import Dict, List
import json
import logging

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, ListField, LabelField
from allennlp.data.fields import ProductionRuleField, MetadataField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset import Dataset
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

    Parameters
    ----------
    tokenizer : ``Tokenizer`` (optional)
        The tokenizer used for sentences in NLVR. Default is ``WordTokenizer``
    sentence_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for tokens in input sentences.
        Default is ``{"tokens": SingleIdTokenIndexer()}``
    nonterminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for non-terminals in production rules.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    terminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for terminals in production rules.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 sentence_token_indexers: Dict[str, TokenIndexer] = None,
                 nonterminal_indexers: Dict[str, TokenIndexer] = None,
                 terminal_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_token_indexers = sentence_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._nonterminal_indexers = nonterminal_indexers or {"tokens":
                                                              SingleIdTokenIndexer("rule_labels")}
        self._terminal_indexers = terminal_indexers or {"tokens": SingleIdTokenIndexer("rule_labels")}
        self._agenda_mapping: Dict[str, str] = {}
        for constant in nlvr_types.COMMON_NAME_MAPPING:
            alias = nlvr_types.COMMON_NAME_MAPPING[constant]
            if alias in nlvr_types.COMMON_TYPE_SIGNATURE:
                constant_type = nlvr_types.COMMON_TYPE_SIGNATURE[alias]
                if constant_type != types.ANY_TYPE:
                    self._agenda_mapping[constant] = "%s -> %s" % (constant_type, constant)

    @overrides
    def read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                instances.append(self.text_to_instance(line))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

    @overrides
    def text_to_instance(self, json_line: str) -> Instance:
        # pylint: disable=arguments-differ
        data = json.loads(json_line)
        sentence = data["sentence"]
        label = data["label"] if "label" in data else None
        structured_rep = data["structured_rep"]
        world = NlvrWorld(structured_rep)
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)
        agenda = self._get_agenda_for_sentence(sentence)
        assert agenda, "No agenda found for sentence: %s" % sentence
        is_nonterminal = lambda x: "<" not in x and len(x) > 1
        production_rule_fields: List[Field] = []
        instance_action_ids: Dict[str, int] = {}
        for production_rule in world.all_possible_actions():
            instance_action_ids[production_rule] = len(instance_action_ids)
            field = ProductionRuleField(production_rule,
                                        terminal_indexers=self._terminal_indexers,
                                        nonterminal_indexers=self._nonterminal_indexers,
                                        is_nonterminal=is_nonterminal,
                                        context=tokenized_sentence)
            production_rule_fields.append(field)
        # agenda_field contains indices into actions.
        agenda_field = ListField([LabelField(instance_action_ids[action],
                                             skip_indexing=True,
                                             label_namespace='actions')
                                  for action in agenda])
        action_field = ListField(production_rule_fields)
        world_field = MetadataField(world)
        fields = {"sentence": sentence_field,
                  "agenda": agenda_field,
                  "world": world_field,
                  "actions": action_field}
        if label:
            label_field = LabelField(label, label_namespace='denotations')
            fields["label"] = label_field
        return Instance(fields)

    def _get_agenda_for_sentence(self, sentence: str) -> List[str]:
        """
        Given a ``sentence``, return a list of actions it triggers. The model tries to include as many
        of these actions in the decoded sequences as possible.
        """
        # TODO(pradeep): Add more rules in the mapping?
        # TODO(pradeep): Use approximate and substring matching as well.
        agenda = ['@START@ -> t']
        # This takes care of shapes, colors, top, bottom, big, small etc.
        for constant, production in self._agenda_mapping.items():
            if constant in sentence:
                agenda.append(production)
        if sentence.startswith("There is a "):
            agenda.append(self._agenda_mapping["assert_greater_equals"])
        if "tower" in sentence or "box" in sentence or "grey" in sentence:
            # Ensuring box filtering function (filter_*) at top.
            agenda.append("t -> [<b,t>, b]")
        return agenda

    @classmethod
    def from_params(cls, params: Params) -> 'NlvrDatasetReader':
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        sentence_token_indexers = TokenIndexer.dict_from_params(params.pop('sentence_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return NlvrDatasetReader(tokenizer=tokenizer,
                                 sentence_token_indexers=sentence_token_indexers)
