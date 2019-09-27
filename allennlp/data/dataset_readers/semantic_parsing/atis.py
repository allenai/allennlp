import json
from typing import Dict, List
import logging
from copy import deepcopy

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
from allennlp.semparse.contexts.atis_sql_table_context import NUMERIC_NONTERMINALS

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

END_OF_UTTERANCE_TOKEN = "@@EOU@@"

def _lazy_parse(text: str):
    for interaction in text.split("\n"):
        if interaction:
            yield json.loads(interaction)

@DatasetReader.register("atis")
class AtisDatasetReader(DatasetReader):
    # pylint: disable=line-too-long
    """
    This ``DatasetReader`` takes json files and converts them into ``Instances`` for the
    ``AtisSemanticParser``.

    Each line in the file is a JSON object that represent an interaction in the ATIS dataset
    that has the following keys and values:
    ```
    "id": The original filepath in the LDC corpus
    "interaction": <list where each element represents a turn in the interaction>
    "scenario": A code that refers to the scenario that served as the prompt for this interaction
    "ut_date": Date of the interaction
    "zc09_path": Path that was used in the original paper `Learning Context-Dependent Mappings from
    Sentences to Logical Form
    <https://www.semanticscholar.org/paper/Learning-Context-Dependent-Mappings-from-Sentences-Zettlemoyer-Collins/44a8fcee0741139fa15862dc4b6ce1e11444878f>'_ by Zettlemoyer and Collins (ACL/IJCNLP 2009)
    ```

    Each element in the ``interaction`` list has the following keys and values:
    ```
    "utterance": Natural language input
    "sql": A list of SQL queries that the utterance maps to, it could be multiple SQL queries
    or none at all.
    ```

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
    database_file: ``str``, optional
        The directory to find the sqlite database file. We query the sqlite database to find the strings
        that are allowed.
    num_turns_to_concatenate: ``str``, optional
        The number of utterances to concatenate as the conversation context.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 keep_if_unparseable: bool = False,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 database_file: str = None,
                 num_turns_to_concatenate: int = 1) -> None:
        super().__init__(lazy)
        self._keep_if_unparseable = keep_if_unparseable
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter())
        self._database_file = database_file
        self._num_turns_to_concatenate = num_turns_to_concatenate

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path) as atis_file:
            logger.info("Reading ATIS instances from dataset at : %s", file_path)
            for line in _lazy_parse(atis_file.read()):
                utterances = []
                for current_interaction in line['interaction']:
                    if not current_interaction['utterance'] or not current_interaction['sql']:
                        continue
                    utterances.append(current_interaction['utterance'])
                    sql_query_labels = [query for query in current_interaction['sql'].split('\n') if query]
                    instance = self.text_to_instance(deepcopy(utterances), sql_query_labels)
                    if not instance:
                        continue
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         utterances: List[str],
                         sql_query_labels: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        utterances: ``List[str]``, required.
            List of utterances in the interaction, the last element is the current utterance.
        sql_query_labels: ``List[str]``, optional
            The SQL queries that are given as labels during training or validation.
        """
        if self._num_turns_to_concatenate:
            utterances[-1] = f' {END_OF_UTTERANCE_TOKEN} '.join(utterances[-self._num_turns_to_concatenate:])

        utterance = utterances[-1]
        action_sequence: List[str] = []

        if not utterance:
            return None

        world = AtisWorld(utterances=utterances)

        if sql_query_labels:
            # If there are multiple sql queries given as labels, we use the shortest
            # one for training.
            sql_query = min(sql_query_labels, key=len)
            try:
                action_sequence = world.get_action_sequence(sql_query)
            except ParseError:
                action_sequence = []
                logger.debug(f'Parsing error')

        tokenized_utterance = self._tokenizer.tokenize(utterance.lower())
        utterance_field = TextField(tokenized_utterance, self._token_indexers)

        production_rule_fields: List[Field] = []

        for production_rule in world.all_possible_actions():
            nonterminal, _ = production_rule.split(' ->')
            # The whitespaces are not semantically meaningful, so we filter them out.
            production_rule = ' '.join([token for token in production_rule.split(' ') if token != 'ws'])
            field = ProductionRuleField(production_rule, self._is_global_rule(nonterminal))
            production_rule_fields.append(field)

        action_field = ListField(production_rule_fields)
        action_map = {action.rule: i # type: ignore
                      for i, action in enumerate(action_field.field_list)}
        index_fields: List[Field] = []
        world_field = MetadataField(world)
        fields = {'utterance' : utterance_field,
                  'actions' : action_field,
                  'world' : world_field,
                  'linking_scores' : ArrayField(world.linking_scores)}

        if sql_query_labels is not None:
            fields['sql_queries'] = MetadataField(sql_query_labels)
            if self._keep_if_unparseable or action_sequence:
                for production_rule in action_sequence:
                    index_fields.append(IndexField(action_map[production_rule], action_field))
                if not action_sequence:
                    index_fields = [IndexField(-1, action_field)]
                action_sequence_field = ListField(index_fields)
                fields['target_action_sequence'] = action_sequence_field
            else:
                # If we are given a SQL query, but we are unable to parse it, and we do not specify explicitly
                # to keep it, then we will skip the it.
                return None

        return Instance(fields)

    @staticmethod
    def _is_global_rule(nonterminal: str) -> bool:
        if nonterminal in NUMERIC_NONTERMINALS:
            return False
        elif nonterminal.endswith('string'):
            return False
        return True
