import json
from typing import Dict, List
import logging

from overrides import overrides
from parsimonious.exceptions import ParseError

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ArrayField, ListField, IndexField, \
        ProductionRuleField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from allennlp.semparse.worlds.atis_world import AtisWorld

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _lazy_parse(text: str):
    for interaction in text.split("\n"):
        if interaction:
            yield json.loads(interaction)

@DatasetReader.register("atis")
class AtisDatasetReader(DatasetReader):
    """
    This ``DatasetReader`` takes json files and converts them into ``Instances`` for the
    ``AtisSemanticParser``.

    Each line in the file is a JSON object that represent an interaction in the ATIS dataset
    that has the following keys and values:
        id: The original filepath in the LDC corpus
        interaction: A list where each element represents a turn in the interaction
            utterance: Natural language input
            sql: A list of SQL queries that the utterance maps to, it could be multiple SQL queries
                or none at all.
        scenario: A code that refers to the scenario that served as the prompt for this interaction
        ut_date: Date of the interaction
        zc09_path: Path that was used in the original paper `Learning Context-Dependent Mappings from
        Sentences to Logical Form
        <https://www.semanticscholar.org/paper/Learning-Context-Dependent-Mappings-from-Sentences-Zettlemoyer-Collins/44a8fcee0741139fa15862dc4b6ce1e11444878f>'_ by Zettlemoyer and Collins (ACL/IJCNLP 2009)


    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Token indexers for the utterances. Will default to ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use for the utterances. Will default to ``WordTokenizer()`` with Spacy's tagger
        enabled.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path) as atis_file:
            logger.info("Reading ATIS instances from dataset at : %s", file_path)
            for line in _lazy_parse(atis_file.read()):
                utterances = []
                for current_interaction in line['interaction']:
                    utterances.append(current_interaction['utterance'])
                    instance = self.text_to_instance(utterances, current_interaction['sql'])
                    # If we can't
                    if not instance:
                        continue
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         utterances: List[str],
                         sql_query: str = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        utterance: ``str``, required.
            Input utterance
        action_sequence: ``List[str]``, required.
            List of actions for the utterance
        world: ``AtisWorld``
            The world in which this utterance appears in, we store this in a MetadataField.
        """
        utterance = utterances[-1]
        action_sequence: List[str] = []

        world = AtisWorld(utterances)

        try:
            action_sequence = world.get_action_sequence(sql_query)
        except ParseError:
            logger.debug(f'Parsing error')

        if not action_sequence:
            return None

        tokenized_utterance = self._tokenizer.tokenize(utterance.lower())
        utterance_field = TextField(tokenized_utterance, self._token_indexers)

        production_rule_fields: List[Field] = []

        for production_rule in world.all_possible_actions():
            lhs, _ = production_rule.split(' ->')
            is_global_rule = not lhs in ['number', 'string']
            # The whitespaces are not semantically meaningful, so we filter them out.
            production_rule = ' '.join([token for token in production_rule.split(' ') if token != 'ws'])
            field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(field)

        action_field = ListField(production_rule_fields)
        action_map = {action.rule: i # type: ignore
                      for i, action in enumerate(action_field.field_list)}
        index_fields: List[Field] = []

        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], action_field))

        action_sequence_field: List[Field] = []
        action_sequence_field.append(ListField(index_fields))

        world_field = MetadataField(world)
        fields = {'utterance' : utterance_field,
                  'actions' : action_field,
                  'world' : world_field,
                  'target_action_sequence' : ListField(action_sequence_field),
                  'linking_scores' : ArrayField(world.linking_scores)}

        return Instance(fields)
