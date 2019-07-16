"""
Reader for WikitableQuestions (https://github.com/ppasupat/WikiTableQuestions/releases/tag/v1.0.2).
"""

import logging
from typing import Dict, List, Any
import os
import gzip
import tarfile

from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.fields import (Field, TextField, MetadataField, ProductionRuleField,
                                  ListField, IndexField, KnowledgeGraphField)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util as wikitables_util
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.semparse import ParsingError
from allennlp.semparse.contexts import TableQuestionContext
from allennlp.semparse.domain_languages import WikiTablesLanguage


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("wikitables")
class WikiTablesDatasetReader(DatasetReader):
    """
    This ``DatasetReader`` takes WikiTableQuestions ``*.examples`` files and converts them into
    ``Instances`` suitable for use with the ``WikiTablesSemanticParser``.

    The ``*.examples`` files have pointers in them to two other files: a file that contains an
    associated table for each question, and a file that has pre-computed, possible logical forms.
    Because of how the ``DatasetReader`` API works, we need to take base directories for those
    other files in the constructor.

    We initialize the dataset reader with paths to the tables directory and the directory where offline search
    output is stored if you are training. While testing, you can either provide existing table
    filenames or if your question is about a new table, provide the content of the table as a dict
    (See :func:`TableQuestionContext.read_from_json` for the expected format). If you are
    doing the former, you still need to provide a ``tables_directory`` path here.

    We lowercase the question and all table text, because the questions in the data are typically
    all lowercase, anyway.  This makes it so that any live demo that you put up will have questions
    that match the data this was trained on.  Lowercasing the table text makes matching the
    lowercased question text easier.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    tables_directory : ``str``, optional
        Prefix for the path to the directory in which the tables reside. For example,
        ``*.examples`` files contain paths like ``csv/204-csv/590.csv``, and we will use the corresponding
        ``tagged`` files by manipulating the paths in the examples files. This is the directory that
        contains the ``csv`` and ``tagged``  directories.  This is only optional for ``Predictors`` (i.e., in a
        demo), where you're only calling :func:`text_to_instance`.
    offline_logical_forms_directory : ``str``, optional
        Directory that contains all the gzipped offline search output files. We assume the filenames match the
        example IDs (e.g.: ``nt-0.gz``).  This is required for training a model, but not required
        for prediction.
    max_offline_logical_forms : ``int``, optional (default=10)
        We will use the first ``max_offline_logical_forms`` logical forms as our target label.  Only
        applicable if ``offline_logical_forms_directory`` is given.
    keep_if_no_logical_forms : ``bool``, optional (default=False)
        If ``True``, we will keep instances we read that don't have offline search output.  If you want to
        compute denotation accuracy on the full dataset, you should set this to ``True``.
        Otherwise, your accuracy numbers will only reflect the subset of the data that has offline search
        output.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use for the questions. Will default to ``WordTokenizer()`` with Spacy's tagger
        enabled, as we use lemma matches as features for entity linking.
    question_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Token indexers for questions. Will default to ``{"tokens": SingleIdTokenIndexer()}``.
    table_token_indexers : ``Dict[str, TokenIndexer]``, optional
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
    output_agendas : ``bool``, (optional, default=False)
        Should we output agenda fields? This needs to be true if you want to train a coverage based
        parser.
    """
    def __init__(self,
                 lazy: bool = False,
                 tables_directory: str = None,
                 offline_logical_forms_directory: str = None,
                 max_offline_logical_forms: int = 10,
                 keep_if_no_logical_forms: bool = False,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 table_token_indexers: Dict[str, TokenIndexer] = None,
                 use_table_for_vocab: bool = False,
                 max_table_tokens: int = None,
                 output_agendas: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._tables_directory = tables_directory
        self._offline_logical_forms_directory = offline_logical_forms_directory
        self._max_offline_logical_forms = max_offline_logical_forms
        self._keep_if_no_logical_forms = keep_if_no_logical_forms
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._table_token_indexers = table_token_indexers or self._question_token_indexers
        self._use_table_for_vocab = use_table_for_vocab
        self._max_table_tokens = max_table_tokens
        self._output_agendas = output_agendas

    @overrides
    def _read(self, file_path: str):
        # Checking if there is a single tarball with all the logical forms. If so, untaring it
        # first.
        if self._offline_logical_forms_directory:
            tarball_with_all_lfs: str = None
            for filename in os.listdir(self._offline_logical_forms_directory):
                if filename.endswith(".tar.gz"):
                    tarball_with_all_lfs = os.path.join(self._offline_logical_forms_directory,
                                                        filename)
                    break
            if tarball_with_all_lfs is not None:
                logger.info(f"Found a tarball in offline logical forms directory: {tarball_with_all_lfs}")
                logger.info("Assuming it contains logical forms for all questions and un-taring it.")
                # If you're running this with beaker, the input directory will be read-only and we
                # cannot untar the files in the directory itself. So we will do so in /tmp, but that
                # means the new offline logical forms directory will be /tmp.
                self._offline_logical_forms_directory = "/tmp/"
                tarfile.open(tarball_with_all_lfs,
                             mode='r:gz').extractall(path=self._offline_logical_forms_directory)
        with open(file_path, "r") as data_file:
            num_missing_logical_forms = 0
            num_lines = 0
            num_instances = 0
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                num_lines += 1
                parsed_info = wikitables_util.parse_example_line(line)
                question = parsed_info["question"]
                # We want the tagged file, but the ``*.examples`` files typically point to CSV.
                table_filename = os.path.join(self._tables_directory,
                                              parsed_info["table_filename"].replace("csv", "tagged"))
                if self._offline_logical_forms_directory:
                    logical_forms_filename = os.path.join(self._offline_logical_forms_directory,
                                                          parsed_info["id"] + '.gz')
                    try:
                        logical_forms_file = gzip.open(logical_forms_filename)
                        logical_forms = []
                        for logical_form_line in logical_forms_file:
                            logical_forms.append(logical_form_line.strip().decode('utf-8'))
                    except FileNotFoundError:
                        logger.debug(f'Missing search output for instance {parsed_info["id"]}; skipping...')
                        logical_forms = None
                        num_missing_logical_forms += 1
                        if not self._keep_if_no_logical_forms:
                            continue
                else:
                    logical_forms = None

                table_lines = [line.split("\t") for line in open(table_filename).readlines()]
                instance = self.text_to_instance(question=question,
                                                 table_lines=table_lines,
                                                 target_values=parsed_info["target_values"],
                                                 offline_search_output=logical_forms)
                if instance is not None:
                    num_instances += 1
                    yield instance

        if self._offline_logical_forms_directory:
            logger.info(f"Missing logical forms for {num_missing_logical_forms} out of {num_lines} instances")
            logger.info(f"Kept {num_instances} instances")

    def text_to_instance(self,  # type: ignore
                         question: str,
                         table_lines: List[List[str]],
                         target_values: List[str] = None,
                         offline_search_output: List[str] = None) -> Instance:
        """
        Reads text inputs and makes an instance. We pass the ``table_lines`` to ``TableQuestionContext``, and that
        method accepts this field either as lines from CoreNLP processed tagged files that come with the dataset,
        or simply in a tsv format where each line corresponds to a row and the cells are tab-separated.

        Parameters
        ----------
        question : ``str``
            Input question
        table_lines : ``List[List[str]]``
            The table content optionally preprocessed by CoreNLP. See ``TableQuestionContext.read_from_lines``
            for the expected format.
        target_values : ``List[str]``, optional
            Target values for the denotations the logical forms should execute to. Not required for testing.
        offline_search_output : ``List[str]``, optional
            List of logical forms, produced by offline search. Not required during test.
        """
        # pylint: disable=arguments-differ
        tokenized_question = self._tokenizer.tokenize(question.lower())
        question_field = TextField(tokenized_question, self._question_token_indexers)
        metadata: Dict[str, Any] = {"question_tokens": [x.text for x in tokenized_question]}
        table_context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
        world = WikiTablesLanguage(table_context)
        world_field = MetadataField(world)
        # Note: Not passing any featre extractors when instantiating the field below. This will make
        # it use all the available extractors.
        table_field = KnowledgeGraphField(table_context.get_table_knowledge_graph(),
                                          tokenized_question,
                                          self._table_token_indexers,
                                          tokenizer=self._tokenizer,
                                          include_in_vocab=self._use_table_for_vocab,
                                          max_table_tokens=self._max_table_tokens)
        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_productions():
            _, rule_right_side = production_rule.split(' -> ')
            is_global_rule = not world.is_instance_specific_entity(rule_right_side)
            field = ProductionRuleField(production_rule, is_global_rule=is_global_rule)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        fields = {'question': question_field,
                  'metadata': MetadataField(metadata),
                  'table': table_field,
                  'world': world_field,
                  'actions': action_field}

        if target_values is not None:
            target_values_field = MetadataField(target_values)
            fields['target_values'] = target_values_field

        # We'll make each target action sequence a List[IndexField], where the index is into
        # the action list we made above.  We need to ignore the type here because mypy doesn't
        # like `action.rule` - it's hard to tell mypy that the ListField is made up of
        # ProductionRuleFields.
        action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore
        if offline_search_output:
            action_sequence_fields: List[Field] = []
            for logical_form in offline_search_output:
                try:
                    action_sequence = world.logical_form_to_action_sequence(logical_form)
                    index_fields: List[Field] = []
                    for production_rule in action_sequence:
                        index_fields.append(IndexField(action_map[production_rule], action_field))
                    action_sequence_fields.append(ListField(index_fields))
                except ParsingError as error:
                    logger.debug(f'Parsing error: {error.message}, skipping logical form')
                    logger.debug(f'Question was: {question}')
                    logger.debug(f'Logical form was: {logical_form}')
                    logger.debug(f'Table info was: {table_lines}')
                    continue
                except KeyError as error:
                    logger.debug(f'Missing production rule: {error.args}, skipping logical form')
                    logger.debug(f'Question was: {question}')
                    logger.debug(f'Table info was: {table_lines}')
                    logger.debug(f'Logical form was: {logical_form}')
                    continue
                except:
                    logger.error(logical_form)
                    raise
                if len(action_sequence_fields) >= self._max_offline_logical_forms:
                    break

            if not action_sequence_fields:
                # This is not great, but we're only doing it when we're passed logical form
                # supervision, so we're expecting labeled logical forms, but we can't actually
                # produce the logical forms.  We should skip this instance.  Note that this affects
                # _dev_ and _test_ instances, too, so your metrics could be over-estimates on the
                # full test data.
                return None
            fields['target_action_sequences'] = ListField(action_sequence_fields)
        if self._output_agendas:
            agenda_index_fields: List[Field] = []
            for agenda_string in world.get_agenda(conservative=True):
                agenda_index_fields.append(IndexField(action_map[agenda_string], action_field))
            if not agenda_index_fields:
                agenda_index_fields = [IndexField(-1, action_field)]
            fields['agenda'] = ListField(agenda_index_fields)
        return Instance(fields)
