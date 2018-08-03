import json
from typing import Dict, List
import logging

from copy import deepcopy
from overrides import overrides
from parsimonious.exceptions import ParseError

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ArrayField, ListField, IndexField, ProductionRuleField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from allennlp.semparse.worlds.atis_world import AtisWorld
from allennlp.semparse.worlds.world import ParsingError
from parsimonious.exceptions import ParseError


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def lazy_parse(text: str):
    for interaction in text.split("\n"):
        if interaction:
            yield json.loads(interaction)

@DatasetReader.register("atis")
class AtisDatasetReader(DatasetReader):
    """
    This ``DatasetReader`` takes json files and converts them into ``Instances`` for the
    ``AtisSemanticParser``.
    """
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
            for line in lazy_parse(atis_file.read()):
                utterances = []
                for current_idx, current_interaction in enumerate(line['interaction']):
                    nl_key = 'utterance'
                    if nl_key not in current_interaction: 
                        nl_key = 'nl_with_dates'

                    utterances.append(current_interaction[nl_key])
                    world = AtisWorld(utterances)

                    try:
                        action_sequence = world.get_action_sequence(current_interaction['sql'])
                        
                    except ParseError as error:
                        # logger.debug(f'Parsing error: {error.message}')
                        logger.debug(f'Parsing error: ')
                        continue

                    instance = self.text_to_instance(current_interaction[nl_key], action_sequence, world)
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
        utterance: ``str``, required.
            Input utterance
        action_sequence: ``List[str]``, required.
            List of actions for the utterance
        world: ``AtisWorld``
            The world in which this utterance appears in, we take the valid actions
            from the world and save it as a MetadataField in the instance.
        """
        tokenized_utterance = self._tokenizer.tokenize(utterance.lower())
        utterance_field = TextField(tokenized_utterance, self._token_indexers)

        production_rule_fields: List[Field] = []

        for production_rule in world.all_possible_actions():
            lhs, _ = production_rule.split(' ->')
            is_global_rule = not lhs in ['number', 'string']
            field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(field)

        action_field = ListField(production_rule_fields)
        action_map = {action.rule.replace(" ws", "").replace("ws ", "") : i # type: ignore
                      for i, action in enumerate(action_field.field_list)}  
        index_fields: List[Field] = []

        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], action_field))

        if not action_sequence:
            return None

        action_sequence_field = ListField(index_fields)

        world_field = MetadataField(world)
        

        fields = {'utterance' : utterance_field,
                  'actions' : action_field,
                  'world' : world_field,
                  'target_action_sequence' : action_sequence_field, 
                  'linking_scores' : ArrayField(world.linking_scores)
                  }

        return Instance(fields)
