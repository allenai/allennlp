"""
Reader for WikitableQuestions (https://github.com/ppasupat/WikiTableQuestions/releases/tag/v1.0.2).
"""

from typing import Dict, List
import logging
import json

from overrides import overrides

from allennlp.common import Params
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, KnowledgeGraphField, ListField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.worlds import WikiTablesWorld

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
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    question_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for questions. Will default to ``{"tokens": SingleIdTokenIndexer()}``.
    table_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for table entities. Will default to ``question_token_indexers`` (though you
        very likely want to use something different for these, as you can't rely on having an
        embedding for every table entity at test time).
    use_table_for_vocab : ``bool`` (optional, default=False)
        If ``True``, we will include table cell text in vocabulary creation.  The original parser
        did not do this, because the same table can appear multiple times, messing with vocab
        counts, and making you include lots of rare entities in your vocab.
    max_table_tokens : ``int``, optional
        If given, we will only keep this number of total table tokens.  This bounds the memory
        usage of the table representations, truncating cells with really long text.  We specify a
        total number of tokens, not a max cell text length, because the number of table entities
        varies.
    """
    def __init__(self,
                 lazy: bool = False,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 table_token_indexers: Dict[str, TokenIndexer] = None,
                 use_table_for_vocab: bool = False,
                 max_table_tokens: int = None) -> None:
        super().__init__(lazy)
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._table_token_indexers = table_token_indexers or self._question_token_indexers
        self._use_table_for_vocab = use_table_for_vocab
        self._max_table_tokens = max_table_tokens

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as data_file:
            for line in data_file.readlines():
                json_obj = json.loads(line)
                yield self.text_to_instance(json_obj)

    @overrides
    def text_to_instance(self, json_obj: JsonDict) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        question_tokens = self._read_tokens_from_json_list(json_obj['question_tokens'])
        question_field = TextField(question_tokens, self._question_token_indexers)
        table_knowledge_graph = TableQuestionKnowledgeGraph.read_from_lines(json_obj['table_lines'],
                                                                            question_tokens)
        entity_tokens = [self._read_tokens_from_json_list(token_list)
                         for token_list in json_obj['entity_texts']]
        table_field = KnowledgeGraphField(table_knowledge_graph,
                                          question_tokens,
                                          tokenizer=None,
                                          token_indexers=self._table_token_indexers,
                                          entity_tokens=entity_tokens,
                                          linking_features=json_obj['linking_features'],
                                          include_in_vocab=self._use_table_for_vocab,
                                          max_table_tokens=self._max_table_tokens)
        world = WikiTablesWorld(table_knowledge_graph)
        world_field = MetadataField(world)

        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_actions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not world.is_table_entity(rule_right_side)
            field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore

        fields = {'question': question_field,
                  'table': table_field,
                  'world': world_field,
                  'actions': action_field}

        if 'target_action_sequences' in json_obj:
            action_sequence_fields: List[Field] = []
            for sequence in json_obj['target_action_sequences']:
                index_fields: List[Field] = []
                for production_rule in sequence:
                    index_fields.append(IndexField(action_map[production_rule], action_field))
                action_sequence_fields.append(ListField(index_fields))
            fields['target_action_sequences'] = ListField(action_sequence_fields)

        if 'example_lisp_string' in json_obj:
            fields['example_lisp_string'] = MetadataField(json_obj['example_lisp_string'])
        elif 'example_string' in json_obj:
            # This is here only for backwards compatibility.
            fields['example_lisp_string'] = MetadataField(json_obj['example_string'])

        return Instance(fields)

    @staticmethod
    def _read_tokens_from_json_list(json_list) -> List[Token]:
        return [Token(text=json_obj['text'], lemma=json_obj['lemma']) for json_obj in json_list]

    @classmethod
    def from_params(cls, params: Params) -> 'WikiTablesPreprocessedDatasetReader':
        lazy = params.pop('lazy', False)
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        table_token_indexers = TokenIndexer.dict_from_params(params.pop('table_token_indexers', {}))
        use_table_for_vocab = params.pop('use_table_for_vocab', False)
        max_table_tokens = params.pop_int('max_table_tokens', None)
        params.assert_empty(cls.__name__)
        return WikiTablesPreprocessedDatasetReader(lazy=lazy,
                                                   question_token_indexers=question_token_indexers,
                                                   table_token_indexers=table_token_indexers,
                                                   use_table_for_vocab=use_table_for_vocab,
                                                   max_table_tokens=max_table_tokens)
