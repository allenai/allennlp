"""
Reader for RateCalculusQuestions.
"""

from typing import Dict, List
import logging
import json

from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, KnowledgeGraphField, ListField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.semparse import ParsingError
from allennlp.semparse.knowledge_graphs import QuestionKnowledgeGraph
from allennlp.semparse.type_declarations import wikitables_type_declaration as wt_types
from allennlp.semparse.worlds import RateCalculusWorld
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ratecalculus")
class RateCalculusDatasetReader(DatasetReader):
    """
    This ``DatasetReader`` takes RateCalculusQuestions files and converts them into
    ``Instances`` suitable for use with the ``SemanticParser``.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 max_dpd_logical_forms: int = 10,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 nonterminal_indexers: Dict[str, TokenIndexer] = None,
                 terminal_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._max_dpd_logical_forms = max_dpd_logical_forms
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._nonterminal_indexers = nonterminal_indexers or {"tokens": SingleIdTokenIndexer("rule_labels")}
        self._terminal_indexers = terminal_indexers or {"token_characters": TokenCharactersIndexer()}
        self._basic_types = set(str(type_) for type_ in wt_types.BASIC_TYPES)

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'r') as f:
            data = f.read().replace('\r\n', '').replace('\n', '').replace('\t', '')
            questions = json.loads(data)

            for q in list(questions):
                sem = q["lSemantics"]
                if( sem == "" ):
                    sem = "(Equals p )"
                instance = self.text_to_instance(q["sQuestion"], [sem])
                if instance is not None:
                    yield instance


    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         dpd_output: List[str] = None,
                         tokenized_question: List[Token] = None) -> Instance:
        """
        Reads text inputs and makes an instance. WikitableQuestions dataset provides tables as TSV
        files, which we use for training. For running a demo, we may want to provide tables in a
        JSON format. To make this method compatible with both, we take ``table_info``, which can
        either be a filename, or a dict. We check the argument's type and call the appropriate
        method in ``TableQuestionKnowledgeGraph``.

        Parameters
        ----------
        question : ``str``
            Input question
        dpd_output : List[str], optional
            List of logical forms, produced by dynamic programming on denotations. Not required
            during test.
        tokenized_question : ``List[Token]``
            If you have already tokenized the question, you can pass that in here, so we don't
            duplicate that work.  You might, for example, do batch processing on the questions in
            the whole dataset, then pass the result in here.
        """
        # pylint: disable=arguments-differ
        tokenized_question = tokenized_question or self._tokenizer.tokenize(question.lower())
        question_field = TextField(tokenized_question, self._question_token_indexers)
        question_knowledge_graph = QuestionKnowledgeGraph.read(tokenized_question)
        question_knowledge_graph_field = KnowledgeGraphField(question_knowledge_graph,
                                          tokenized_question,
                                          self._question_token_indexers,
                                          tokenizer=self._tokenizer)

        world = RateCalculusWorld(question_knowledge_graph)
        world_field = MetadataField(world)

        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_actions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not world.is_question_entity(rule_right_side)
            field = ProductionRuleField(production_rule, is_global_rule)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        fields = {'question': question_field,
                  'question_knowledge_graph': question_knowledge_graph_field,
                  'world': world_field,
                  'actions': action_field}

        if dpd_output:
            # We'll make each target action sequence a List[IndexField], where the index is into
            # the action list we made above.  We need to ignore the type here because mypy doesn't
            # like `action.rule` - it's hard to tell mypy that the ListField is made up of
            # ProductionRuleFields.
            action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore

            action_sequence_fields: List[Field] = []
            for logical_form in dpd_output:
                try:
                    expression = world.parse_logical_form(logical_form)
                except ParsingError as error:
                    logger.debug(f'Parsing error: {error.message}, skipping logical form')
                    logger.debug(f'Question was: {question}')
                    logger.debug(f'Logical form was: {logical_form}')
                    continue
                except:
                    logger.error(logical_form)
                    raise
                action_sequence = world.get_action_sequence(expression)
                try:
                    index_fields: List[Field] = []
                    for production_rule in action_sequence:
                        index_fields.append(IndexField(action_map[production_rule], action_field))
                    action_sequence_fields.append(ListField(index_fields))
                except KeyError as error:
                    logger.debug(f'Missing production rule: {error.args}, skipping logical form')
                    logger.debug(f'Question was: {question}')
                    logger.debug(f'Logical form was: {logical_form}')
                    continue
                if len(action_sequence_fields) >= self._max_dpd_logical_forms:
                    break

            if not action_sequence_fields:
                # This is not great, but we're only doing it when we're passed logical form
                # supervision, so we're expecting labeled logical forms, but we can't actually
                # produce the logical forms.  We should skip this instance.  Note that this affects
                # _dev_ and _test_ instances, too, so your metrics could be over-estimates on the
                # full test data.
                return None

            fields['target_action_sequences'] = ListField(action_sequence_fields)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'RateCalculusDatasetReader':
        lazy = params.pop('lazy', False)
        max_dpd_logical_forms = params.pop_int('max_dpd_logical_forms', 10)
        default_tokenizer_params = {'word_splitter': {'type': 'spacy', 'pos_tags': True}}
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', default_tokenizer_params))
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return RateCalculusDatasetReader(lazy=lazy,
                                       max_dpd_logical_forms=max_dpd_logical_forms,
                                       tokenizer=tokenizer,
                                       question_token_indexers=question_token_indexers,
                                         nonterminal_indexers = None,
                                         terminal_indexers = None)
