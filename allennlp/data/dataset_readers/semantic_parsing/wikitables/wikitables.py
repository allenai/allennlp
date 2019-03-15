"""
Reader for WikitableQuestions (https://github.com/ppasupat/WikiTableQuestions/releases/tag/v1.0.2).
"""

from typing import Dict, List, Any
import gzip
import json
import logging
import os

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, IndexField, KnowledgeGraphField, ListField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph
from allennlp.semparse.type_declarations import wikitables_lambda_dcs as wt_types
from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.semparse.worlds.world import ParsingError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("wikitables")
class WikiTablesDatasetReader(DatasetReader):
    """
    This ``DatasetReader`` takes WikiTableQuestions ``*.examples`` files and converts them into
    ``Instances`` suitable for use with the ``WikiTablesSemanticParser``.  This reader also accepts
    pre-processed JSONL files produced by ``scripts/preprocess_wikitables.py``.  Processing the
    example files to read a bunch of individual table files, run NER on all of the tables, and
    convert logical forms into action sequences is quite slow, so we recommend you run the
    pre-processing script.  Having the same reader handle both file types allows you to train with
    a pre-processed file, but not have to change your model configuration in order to serve a demo
    from the trained model.

    The ``*.examples`` files have pointers in them to two other files: a file that contains an
    associated table for each question, and a file that has pre-computed, possible logical forms.
    Because of how the ``DatasetReader`` API works, we need to take base directories for those
    other files in the constructor.

    We initialize the dataset reader with paths to the tables directory and the directory where DPD
    output is stored if you are training. While testing, you can either provide existing table
    filenames or if your question is about a new table, provide the content of the table as a dict
    (See :func:`TableQuestionKnowledgeGraph.read_from_json` for the expected format). If you are
    doing the former, you still need to provide a ``tables_directory`` path here.

    For training, we assume you are reading in ``data/*.examples`` files, and you have access to
    the output from Dynamic Programming on Denotations (DPD) on the training dataset.

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
        ``*.examples`` files contain paths like ``csv/204-csv/590.csv``, this is the directory that
        contains the ``csv`` directory.  This is only optional for ``Predictors`` (i.e., in a
        demo), where you're only calling :func:`text_to_instance`.
    dpd_output_directory : ``str``, optional
        Directory that contains all the gzipped dpd output files. We assume the filenames match the
        example IDs (e.g.: ``nt-0.gz``).  This is required for training a model, but not required
        for prediction.
    max_dpd_logical_forms : ``int``, optional (default=10)
        We will use the first ``max_dpd_logical_forms`` logical forms as our target label.  Only
        applicable if ``dpd_output_directory`` is given.
    sort_dpd_logical_forms : ``bool``, optional (default=True)
        If ``True``, we will sort the logical forms in the DPD output by length before selecting
        the first ``max_dpd_logical_forms``.  This makes the data loading quite a bit slower, but
        results in better training data.
    max_dpd_tries : ``int``, optional
        Sometimes DPD just made bad choices about logical forms and gives us forms that we can't
        parse (most of the time these are very unlikely logical forms, because, e.g., it
        hallucinates a date or number from the table that's not in the question).  But we don't
        want to spend our time trying to parse thousands of bad logical forms.  We will try to
        parse only the first ``max_dpd_tries`` logical forms before giving up.  This also speeds up
        data loading time, because we don't go through the entire DPD file if it's huge (unless
        we're sorting the logical forms).  Only applicable if ``dpd_output_directory`` is given.
        Default is 20.
    keep_if_no_dpd : ``bool``, optional (default=False)
        If ``True``, we will keep instances we read that don't have DPD output.  If you want to
        compute denotation accuracy on the full dataset, you should set this to ``True``.
        Otherwise, your accuracy numbers will only reflect the subset of the data that has DPD
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
    linking_feature_extractors : ``List[str]``, optional
        The list of feature extractors to use in the :class:`KnowledgeGraphField` when computing
        entity linking features.  See that class for more information.  By default, we will use all
        available feature extractors.
    include_table_metadata : ``bool`` (optional, default=False)
        This is necessary for pre-processing the data.  We output a jsonl file that has all of the
        information necessary for reading each instance, which includes the table contents itself.
        This flag tells the reader to include a ``table_metadata`` field that gets read by the
        pre-processing script.
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
                 dpd_output_directory: str = None,
                 max_dpd_logical_forms: int = 10,
                 sort_dpd_logical_forms: bool = True,
                 max_dpd_tries: int = 20,
                 keep_if_no_dpd: bool = False,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 table_token_indexers: Dict[str, TokenIndexer] = None,
                 use_table_for_vocab: bool = False,
                 linking_feature_extractors: List[str] = None,
                 include_table_metadata: bool = False,
                 max_table_tokens: int = None,
                 output_agendas: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._tables_directory = tables_directory
        self._dpd_output_directory = dpd_output_directory
        self._max_dpd_logical_forms = max_dpd_logical_forms
        self._sort_dpd_logical_forms = sort_dpd_logical_forms
        self._max_dpd_tries = max_dpd_tries
        self._keep_if_no_dpd = keep_if_no_dpd
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter(pos_tags=True))
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._table_token_indexers = table_token_indexers or self._question_token_indexers
        self._use_table_for_vocab = use_table_for_vocab
        self._linking_feature_extractors = linking_feature_extractors
        self._include_table_metadata = include_table_metadata
        self._basic_types = set(str(type_) for type_ in wt_types.BASIC_TYPES)
        self._max_table_tokens = max_table_tokens
        self._output_agendas = output_agendas

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith('.examples'):
            yield from self._read_examples_file(file_path)
        elif file_path.endswith('.jsonl'):
            yield from self._read_preprocessed_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        with open(file_path, "r") as data_file:
            num_dpd_missing = 0
            num_lines = 0
            num_instances = 0
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                num_lines += 1
                parsed_info = util.parse_example_line(line)
                question = parsed_info["question"]
                # We want the TSV file, but the ``*.examples`` files typically point to CSV.
                table_filename = os.path.join(self._tables_directory,
                                              parsed_info["table_filename"].replace(".csv", ".tsv"))
                if self._dpd_output_directory:
                    dpd_output_filename = os.path.join(self._dpd_output_directory,
                                                       parsed_info["id"] + '.gz')
                    try:
                        dpd_file = gzip.open(dpd_output_filename)
                        if self._sort_dpd_logical_forms:
                            sempre_forms = [dpd_line.strip().decode('utf-8') for dpd_line in dpd_file]
                            # We'll sort by the number of open parens in the logical form, which
                            # tells you how many nodes there are in the syntax tree.
                            sempre_forms.sort(key=lambda x: x.count('('))
                            if self._max_dpd_tries:
                                sempre_forms = sempre_forms[:self._max_dpd_tries]
                        else:
                            sempre_forms = []
                            for dpd_line in dpd_file:
                                sempre_forms.append(dpd_line.strip().decode('utf-8'))
                                if self._max_dpd_tries and len(sempre_forms) >= self._max_dpd_tries:
                                    break
                    except FileNotFoundError:
                        logger.debug(f'Missing DPD output for instance {parsed_info["id"]}; skipping...')
                        sempre_forms = None
                        num_dpd_missing += 1
                        if not self._keep_if_no_dpd:
                            continue
                else:
                    sempre_forms = None

                table_lines = open(table_filename).readlines()
                instance = self.text_to_instance(question=question,
                                                 table_lines=table_lines,
                                                 example_lisp_string=line,
                                                 dpd_output=sempre_forms)
                if instance is not None:
                    num_instances += 1
                    yield instance

        if self._dpd_output_directory:
            logger.info(f"Missing DPD info for {num_dpd_missing} out of {num_lines} instances")
            num_with_dpd = num_lines - num_dpd_missing
            num_bad_lfs = num_with_dpd - num_instances
            logger.info(f"DPD output was bad for {num_bad_lfs} out of {num_with_dpd} instances")
            if num_bad_lfs > 0:
                logger.info("Re-run with log level set to debug to see the un-parseable logical forms")
            logger.info(f"Kept {num_instances} instances")

    def _read_preprocessed_file(self, file_path: str):
        with open(file_path, "r") as data_file:
            for line in data_file.readlines():
                json_obj = json.loads(line)
                yield self._json_blob_to_instance(json_obj)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         table_lines: List[str],
                         example_lisp_string: str = None,
                         dpd_output: List[str] = None,
                         tokenized_question: List[Token] = None) -> Instance:
        """
        Reads text inputs and makes an instance. WikitableQuestions dataset provides tables as TSV
        files, which we use for training.

        Parameters
        ----------
        question : ``str``
            Input question
        table_lines : ``List[str]``
            The table content itself, as a list of rows. See
            ``TableQuestionKnowledgeGraph.read_from_lines`` for the expected format.
        example_lisp_string : ``str``, optional
            The original (lisp-formatted) example string in the WikiTableQuestions dataset.  This
            comes directly from the ``.examples`` file provided with the dataset.  We pass this to
            SEMPRE for evaluating logical forms during training.  It isn't otherwise used for
            anything.
        dpd_output : List[str], optional
            List of logical forms, produced by dynamic programming on denotations. Not required
            during test.
        tokenized_question : ``List[Token]``, optional
            If you have already tokenized the question, you can pass that in here, so we don't
            duplicate that work.  You might, for example, do batch processing on the questions in
            the whole dataset, then pass the result in here.
        """
        # pylint: disable=arguments-differ
        tokenized_question = tokenized_question or self._tokenizer.tokenize(question.lower())
        question_field = TextField(tokenized_question, self._question_token_indexers)
        metadata: Dict[str, Any] = {"question_tokens": [x.text for x in tokenized_question]}
        metadata["original_table"] = "\n".join(table_lines)
        table_knowledge_graph = TableQuestionKnowledgeGraph.read_from_lines(table_lines, tokenized_question)
        table_metadata = MetadataField(table_lines)
        table_field = KnowledgeGraphField(table_knowledge_graph,
                                          tokenized_question,
                                          self._table_token_indexers,
                                          tokenizer=self._tokenizer,
                                          feature_extractors=self._linking_feature_extractors,
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

        fields = {'question': question_field,
                  'metadata': MetadataField(metadata),
                  'table': table_field,
                  'world': world_field,
                  'actions': action_field}
        if self._include_table_metadata:
            fields['table_metadata'] = table_metadata
        if example_lisp_string:
            fields['example_lisp_string'] = MetadataField(example_lisp_string)

        # We'll make each target action sequence a List[IndexField], where the index is into
        # the action list we made above.  We need to ignore the type here because mypy doesn't
        # like `action.rule` - it's hard to tell mypy that the ListField is made up of
        # ProductionRuleFields.
        action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore
        if dpd_output:
            action_sequence_fields: List[Field] = []
            for logical_form in dpd_output:
                if not self._should_keep_logical_form(logical_form):
                    logger.debug(f'Question was: {question}')
                    logger.debug(f'Table info was: {table_lines}')
                    continue
                try:
                    expression = world.parse_logical_form(logical_form)
                except ParsingError as error:
                    logger.debug(f'Parsing error: {error.message}, skipping logical form')
                    logger.debug(f'Question was: {question}')
                    logger.debug(f'Logical form was: {logical_form}')
                    logger.debug(f'Table info was: {table_lines}')
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
                    logger.debug(f'Table info was: {table_lines}')
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
        if self._output_agendas:
            agenda_index_fields: List[Field] = []
            for agenda_string in world.get_agenda():
                agenda_index_fields.append(IndexField(action_map[agenda_string], action_field))
            if not agenda_index_fields:
                agenda_index_fields = [IndexField(-1, action_field)]
            fields['agenda'] = ListField(agenda_index_fields)
        return Instance(fields)

    def _json_blob_to_instance(self, json_obj: JsonDict) -> Instance:
        question_tokens = self._read_tokens_from_json_list(json_obj['question_tokens'])
        question_field = TextField(question_tokens, self._question_token_indexers)
        question_metadata = MetadataField({"question_tokens": [x.text for x in question_tokens]})
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

        example_string_field = MetadataField(json_obj['example_lisp_string'])

        fields = {'question': question_field,
                  'metadata': question_metadata,
                  'table': table_field,
                  'world': world_field,
                  'actions': action_field,
                  'example_lisp_string': example_string_field}

        if 'target_action_sequences' in json_obj or 'agenda' in json_obj:
            action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore
        if 'target_action_sequences' in json_obj:
            action_sequence_fields: List[Field] = []
            for sequence in json_obj['target_action_sequences']:
                index_fields: List[Field] = []
                for production_rule in sequence:
                    index_fields.append(IndexField(action_map[production_rule], action_field))
                action_sequence_fields.append(ListField(index_fields))
            fields['target_action_sequences'] = ListField(action_sequence_fields)
        if 'agenda' in json_obj:
            agenda_index_fields: List[Field] = []
            for agenda_action in json_obj['agenda']:
                agenda_index_fields.append(IndexField(action_map[agenda_action], action_field))
            fields['agenda'] = ListField(agenda_index_fields)
        return Instance(fields)

    @staticmethod
    def _read_tokens_from_json_list(json_list) -> List[Token]:
        return [Token(text=json_obj['text'], lemma_=json_obj['lemma']) for json_obj in json_list]

    @staticmethod
    def _should_keep_logical_form(logical_form: str) -> bool:
        # DPD has funny ideas about long strings of "ors" being reasonable logical forms.  They
        # aren't, and they crash our recursive type inference code.  TODO(mattg): we need to fix
        # the type inference code to not die in those cases, somehow...
        if logical_form.count('(or') > 3:
            logger.debug(f'Skipping logical form with inordinate number of "ors": {logical_form}')
            return False
        return True
