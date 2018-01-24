"""
Reader for WikitableQuestions (https://github.com/ppasupat/WikiTableQuestions/releases/tag/v1.0.2).
"""

from typing import Dict, List, Union
import logging
import os
import json

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, KnowledgeGraphField, ListField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.semparse import ParsingError
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph
from allennlp.data.semparse.type_declarations import wikitables_type_declaration as wt_types
from allennlp.data.semparse.worlds import WikiTablesWorld
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("wikitables-preprocessed")
class WikiTablesPreprocessedDatasetReader(DatasetReader):
    """
    This ``DatasetReader`` takes a json-lines file output by the ``preprocess_wikitables.py``
    script.  This is much faster for quick iteration on your model.  This pre-computes the target
    action sequences and the linking features; the text representations for the question and table
    entities are still configurable at runtime with this dataset reader.

    Parameters
    ----------
    question_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for questions. Will default to ``{"tokens": SingleIdTokenIndexer()}``.
    table_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for table entities. Will default to ``question_token_indexers`` (though you
        very likely want to use something different for these, as you can't rely on having an
        embedding for every table entity at test time).
    nonterminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        How should we represent non-terminals in production rules when we're computing action
        embeddings?  We use ``TokenIndexers`` for this.  Default is to use a
        ``SingleIdTokenIndexer`` with the ``rule_labels`` namespace, keyed by ``tokens``:
        ``{"tokens": SingleIdTokenIndexer("rule_labels")}``.  We use the namespace ``rule_labels``
        so that we don't get padding or OOV tokens for nonterminals.
    terminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        How should we represent terminals in production rules when we're computing action
        embeddings?  We also use ``TokenIndexers`` for this.  The default is to use a
        ``TokenCharactersIndexer`` keyed by ``token_characters``: ``{"token_characters":
        TokenCharactersIndexer()}``.  We use this indexer by default because WikiTables has plenty
        of terminals that are unseen at training time, so we need to use a representation for them
        that is not just a vocabulary lookup.
    """
    def __init__(self,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 table_token_indexers: Dict[str, TokenIndexer] = None,
                 nonterminal_indexers: Dict[str, TokenIndexer] = None,
                 terminal_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._table_token_indexers = table_token_indexers or self._question_token_indexers
        self._nonterminal_indexers = nonterminal_indexers or {"tokens": SingleIdTokenIndexer("rule_labels")}
        self._terminal_indexers = terminal_indexers or {"token_characters": TokenCharactersIndexer()}

    @overrides
    def read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in tqdm.tqdm(data_file.readlines()):
                json_obj = json.loads(line)
                question_tokens = self._read_tokens_from_json_list(json_obj['question_tokens'])
                question_field = TextField(question_tokens, self._question_token_indexers)
                table_knowledge_graph = TableKnowledgeGraph.read_from_lines(json_obj['table_lines'])
                entity_tokens = [self._read_tokens_from_json_list(token_list)
                                 for token_list in json_obj['entity_texts']]
                table_field = KnowledgeGraphField(table_knowledge_graph,
                                                  question_tokens,
                                                  tokenizer=None,
                                                  token_indexers=self._table_token_indexers,
                                                  entity_tokens=entity_tokens,
                                                  linking_features=json_obj['linking_features'])
                world = WikiTablesWorld(table_knowledge_graph, question_tokens)
                world_field = MetadataField(world)

                production_rule_fields: List[Field] = []
                for production_rule in world.all_possible_actions():
                    field = ProductionRuleField(production_rule,
                                                terminal_indexers=self._terminal_indexers,
                                                nonterminal_indexers=self._nonterminal_indexers,
                                                is_nonterminal=lambda x: not world.is_table_entity(x),
                                                context=question_tokens)
                    production_rule_fields.append(field)
                action_field = ListField(production_rule_fields)

                action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore
                action_sequence_fields: List[Field] = []
                for sequence in json_obj['target_action_sequences']:
                    index_fields: List[Field] = []
                    for production_rule in sequence:
                        index_fields.append(IndexField(action_map[production_rule], action_field))
                    action_sequence_fields.append(ListField(index_fields))
                targets_field = ListField(action_sequence_fields)

                fields = {'question': question_field,
                          'table': table_field,
                          'world': world_field,
                          'actions': action_field,
                          'target_action_sequences': targets_field}
                instances.append(Instance(fields))
        return Dataset(instances)

    @overrides
    def text_to_instance(self) -> Instance:  # type: ignore
        raise NotImplementedError()

    @staticmethod
    def _read_tokens_from_json_list(json_list) -> List[Token]:
        return [Token(text=json_obj['text'], lemma=json_obj['lemma']) for json_obj in json_list]

    @classmethod
    def from_params(cls, params: Params) -> 'WikiTablesDatasetReader':
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        table_token_indexers = TokenIndexer.dict_from_params(params.pop('table_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return WikiTablesDatasetReader(question_token_indexers=question_token_indexers,
                                       table_token_indexers=table_token_indexers)
