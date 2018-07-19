import json
from typing import Dict, Tuple, List
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ListField, IndexField, ProductionRuleField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from allennlp.semparse.contexts.atis_tables import ConversationContext
from allennlp.semparse.worlds.atis_world import AtisWorld 
from allennlp.semparse.worlds.world import ParsingError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def lazy_parse(text: str):
    for interaction in text.split("\n"):
        if interaction:
            yield json.loads(interaction) 


@DatasetReader.register("atis")
class AtisDatasetReader(DatasetReader):
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None
                 ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))

        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path) as atis_file:
            logger.info("Reading ATIS instances from dataset at : %s", file_path)
           
            for interaction in lazy_parse(atis_file.read()):
                conv_context = ConversationContext(interaction['interaction'])
                for interaction_round in conv_context.interaction:
                    nl_key = 'utterance'
                    if nl_key not in interaction_round:
                        nl_key = 'nl_with_dates'

                    world = AtisWorld(conv_context, interaction_round[nl_key])
                    action_sequence = []
                    try:
                        conv_context.valid_actions = world.valid_actions
                        action_sequence = world.get_action_sequence(interaction_round['sql'])
                        
                    except: 
                        print('parsing error')
                        continue

                    print('yield instance')
                    instance = self.text_to_instance(interaction_round[nl_key], action_sequence, world)
                    if not instance:
                        continue
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         utterance: str,
                         action_sequence: List[str],
                         world: AtisWorld
                         ) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        words : ``List[str]``, required.
        action_sequences: ``List[str]``, requred.
        world: ``AtisWorld``

        Returns
        -------
        """
        tokenized_utterance = self._tokenizer.tokenize(utterance.lower())
        utterance_field = TextField(tokenized_utterance, self._token_indexers)

        production_rule_fields: List[Field] = []

        for production_rule in world.all_possible_actions():
            print(production_rule)
            lhs, _ = production_rule.split(' ->')
            is_global_rule = not lhs in ['number', 'string'] 
            field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(field)

        action_field = ListField(production_rule_fields)
        
        action_map = {action.rule.replace(" ws", "").replace("ws ", "") : i for i, action in enumerate(action_field.field_list)}  # type: ignore
        index_fields : List[Field] = []

        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], action_field))
        
        field_class_set = set([field.__class__ for field in index_fields])

        if not action_sequence:
            return None
        

        action_sequence_field = [ListField(index_fields)]

        world_field = MetadataField(world)

        fields = {'utterance' : utterance_field,
                  'actions' : action_field,
                  'world' : world_field,
                  'target_action_sequence' : ListField(action_sequence_field)}

        print('dsr action field', world.all_possible_actions())
        print('dsr valid actions', world.get_valid_actions())
        return Instance(fields)


    @classmethod
    def from_params(cls, params: Params) -> 'AtisDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        
        source_indexers_type = params.pop('source_token_indexers', None)
        if source_indexers_type is None:
            source_token_indexers = None
        else:
            source_token_indexers = TokenIndexer.dict_from_params(source_indexers_type)
        target_indexers_type = params.pop('target_token_indexers', None)
        if target_indexers_type is None:
            target_token_indexers = None
        else:
            target_token_indexers = TokenIndexer.dict_from_params(target_indexers_type)

        params.assert_empty(cls.__name__)
        return AtisDatasetReader(source_token_indexers=source_token_indexers,
                                 target_token_indexers=target_token_indexers,
                                 token_indexers=token_indexers,
                                 lazy=lazy)
