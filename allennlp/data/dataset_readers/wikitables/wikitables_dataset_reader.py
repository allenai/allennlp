"""
Reader for WikitableQuestions (https://github.com/ppasupat/WikiTableQuestions/releases/tag/v1.0.2).
"""

from typing import Dict, List, Union
import gzip
import logging
import os
import pyparsing

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict
from allennlp.data.dataset import Dataset
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.fields import Field, IndexField, KnowledgeGraphField, ListField
from allennlp.data.fields import MetadataField, ProductionRuleField, TextField
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph
from allennlp.data.semparse.type_declarations import wikitables_type_declaration as wt_types
from allennlp.data.semparse.worlds import WikiTablesWorld
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("wikitables")
class WikiTablesDatasetReader(DatasetReader):
    """
    This ``DatasetReader`` takes WikiTableQuestions example files and converts them into
    ``Instances`` suitable for use with the ``WikiTablesSemanticParser``.  The example files have
    pointers in them to two other files: a file that contains an associated table for each
    question, and a file that has pre-computed, possible logical forms.  Because of how the
    ``DatasetReader`` API works, we need to take base directories for those other files in the
    constructor.

    We initialize the dataset reader with paths to the tables directory and the directory where DPD
    output is stored if you are training. While testing, you can either provide existing table
    filenames or if your question is about a new table, provide the content of the table as a dict
    (See :func:`TableKnowledgeGraph.read_from_json` for the expected format). If you are doing the
    former, you still need to provide a ``tables_directory`` path here.

    For training, we assume you are reading in ``data/*.examples`` files, and you have access to
    the output from Dynamic Programming on Denotations (DPD) on the training dataset.

    Parameters
    ----------
    tables_directory : ``str`` (optional)
        Prefix for the path to the directory in which the tables reside. For example,
        ``*.examples`` files contain paths like ``csv/204-csv/590.csv``, this is the directory that
        contains the ``csv`` directory.
    dpd_output_directory : ``str`` (optional)
        Directory that contains all the gzipped dpd output files. We assume the filenames match the
        example IDs (e.g.: ``nt-0.gz``).
    max_dpd_logical_forms : ``int`` (optional)
        We will use the first ``max_dpd_logical_forms`` logical forms as our target label.  Only
        applicable if ``dpd_output_directory`` is given.  Default is 10.
    max_dpd_tries : ``int`` (optional)
        Sometimes DPD just made bad choices about logical forms and gives us forms that we can't
        parse (most of the time these are very unlikely logical forms, because, e.g., it
        halucinates a date or number from the table that's not in the question).  But we don't want
        to spend our time trying to parse thousands of bad logical forms.  We will try at most
        ``max_dpd_tries`` logical forms before giving up.  Only applicable if
        ``dpd_output_directory`` is given.  Default is 20.
    tokenizer : ``Tokenizer`` (optional)
        Tokenizer to use for the questions. Will default to ``WordTokenizer()``.
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
                 tables_directory: str = None,
                 dpd_output_directory: str = None,
                 max_dpd_logical_forms: int = 10,
                 max_dpd_tries: int = 20,
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 table_token_indexers: Dict[str, TokenIndexer] = None,
                 nonterminal_indexers: Dict[str, TokenIndexer] = None,
                 terminal_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tables_directory = tables_directory
        self._dpd_output_directory = dpd_output_directory
        self._max_dpd_logical_forms = max_dpd_logical_forms
        self._max_dpd_tries = max_dpd_tries
        self._tokenizer = tokenizer or WordTokenizer()
        self._question_token_indexers = question_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._table_token_indexers = table_token_indexers or self._question_token_indexers
        self._nonterminal_indexers = nonterminal_indexers or {"tokens": SingleIdTokenIndexer("rule_labels")}
        self._terminal_indexers = terminal_indexers or {"token_characters": TokenCharactersIndexer()}
        self._basic_types = set(str(type_) for type_ in wt_types.BASIC_TYPES)

    @overrides
    def read(self, file_path):
        instances = []
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            num_dpd_missing = 0
            num_lines = 0
            for line in tqdm.tqdm(data_file.readlines()):
                line = line.strip("\n")
                if not line:
                    continue
                num_lines += 1
                parsed_info = self._parse_line_as_lisp(line)
                question = parsed_info["question"]
                # We want the TSV file, but the ``*.examples`` files typically point to CSV.
                table_filename = os.path.join(self._tables_directory,
                                              parsed_info["context"].replace(".csv", ".tsv"))
                dpd_output_filename = os.path.join(self._dpd_output_directory,
                                                   parsed_info["id"] + '.gz')
                try:
                    sempre_forms = [line.strip().decode('utf-8') for line in gzip.open(dpd_output_filename)]
                except FileNotFoundError:
                    logger.debug(f'Missing DPD output for instance {parsed_info["id"]}; skipping...')
                    num_dpd_missing += 1
                    continue
                instance = self.text_to_instance(question, table_filename, sempre_forms)
                if instance is not None:
                    # The DPD output might not actually give us usable logical forms for some
                    # instances, and in those cases `text_to_instance` returns None.
                    instances.append(instance)
        logger.info(f"Missing DPD info for {num_dpd_missing} out of {num_lines} instances")
        num_instances = len(instances)
        num_with_dpd = num_lines - num_dpd_missing
        num_bad_lfs = num_with_dpd - num_instances
        logger.info(f"DPD output was bad for {num_bad_lfs} out of {num_with_dpd} instances")
        if num_bad_lfs > 0:
            logger.info("Re-run with log level set to debug to see the un-parseable logical forms")
        logger.info(f"Kept {num_instances} instances")
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         table_info: Union[str, JsonDict],
                         dpd_output: List[str] = None) -> Instance:
        """
        Reads text inputs and makes an instance. WikitableQuestions dataset provides tables as TSV
        files, which we use for training. For running a demo, we may want to provide tables in a
        JSON format. To make this method compatible with both, we take ``table_info``, which can
        either be a filename, or a dict. We check the argument's type and calle the appropriate
        method in ``TableKnowledgeGraph``.

        Parameters
        ----------
        question : ``str``
            Input question
        table_info : ``str`` or ``JsonDict``
            Table filename or the table content itself, as a dict. See
            ``TableKnowledgeGraph.read_from_json`` for the expected format.
        dpd_output : List[str] (optional)
            List of logical forms, produced by dynamic programming on denotations. Not required
            during test.
        """
        # pylint: disable=arguments-differ
        tokenized_question = self._tokenizer.tokenize(question)
        question_field = TextField(tokenized_question, self._question_token_indexers)
        if isinstance(table_info, str):
            table_knowledge_graph = TableKnowledgeGraph.read_from_file(table_info)
        else:
            table_knowledge_graph = TableKnowledgeGraph.read_from_json(table_info)
        table_field = KnowledgeGraphField(table_knowledge_graph, self._table_token_indexers)
        world = WikiTablesWorld(table_knowledge_graph, tokenized_question)
        world_field = MetadataField(world)

        production_rule_fields: List[Field] = []
        for production_rule in world.all_possible_actions():
            field = ProductionRuleField(production_rule,
                                        terminal_indexers=self._terminal_indexers,
                                        nonterminal_indexers=self._nonterminal_indexers,
                                        nonterminal_types=self._basic_types,
                                        context=tokenized_question)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)

        fields = {'question': question_field,
                  'table': table_field,
                  'world': world_field,
                  'actions': action_field}

        if dpd_output:
            # We'll make each target action sequence a List[IndexField], where the index is into
            # the action list we made above.  We need to ignore the type here because mypy doesn't
            # like `action.rule` - it's hard to tell mypy that the ListField is made up of
            # ProductionRuleFields.
            action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore

            action_sequence_fields: List[Field] = []
            for logical_form in dpd_output[:self._max_dpd_tries]:
                if not self._should_keep_logical_form(logical_form):
                    continue
                try:
                    expression = world.parse_logical_form(logical_form)
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

    @staticmethod
    def _parse_line_as_lisp(lisp_string: str) -> Dict[str, Union[str, List[str], None]]:
        """
        Training data in WikitableQuestions comes with examples in the form of lisp strings in the format:
            (example (id <example-id>)
                     (utterance <question>)
                     (context (graph tables.TableKnowledgeGraph <table-filename>))
                     (targetValue (list (description <answer1>) (description <answer2>) ...)))

        We parse such strings and return the parsed information here.
        """
        parsed_info = {}
        input_nested_list = pyparsing.OneOrMore(pyparsing.nestedExpr()).parseString(lisp_string).asList()[0]
        # Skipping "example"
        for key, value in input_nested_list[1:]:
            if key == "id":
                parsed_info["id"] = value
            elif key == "utterance":
                parsed_info["question"] = value.replace("\"", "")
            elif key == "context":
                # Skipping "graph" and "tables.TableKnowledgeGraph"
                parsed_info["context"] = value[-1]
            elif key == "targetValue":
                # Skipping "list", and "description" within each nested list.
                parsed_info["targetValue"] = [x[1] for x in value[1:]]
        # targetValue may not be present if the answer is not provided.
        assert all([x in parsed_info for x in ["id", "question", "context"]]), "Invalid format"
        return parsed_info

    @staticmethod
    def _should_keep_logical_form(logical_form: str) -> bool:
        # DPD has funny ideas about long strings of "ors" being reasonable logical forms.  They
        # aren't, and they crash our recursive type inference code.  TODO(mattg): we need to fix
        # the type inference code to not die in those cases, somehow...
        if logical_form.count('(or') > 3:
            logger.debug(f'Skipping logical form with inordinate number of "ors": {logical_form}')
            return False
        if 'fb:part' in logical_form:
            # TODO(mattg): we don't currently ever create production rules to generate cell parts,
            # and it's not clear to me why we ever should.  These always fail to parse right now,
            # so we'll just skip them and fix it later.
            logger.debug(f'Skipping logical form with "fb.part": {logical_form}')
            return False
        # TODO(mattg): check for dates here
        return True

    @classmethod
    def from_params(cls, params: Params) -> 'WikiTablesDatasetReader':
        tables_directory = params.pop('tables_directory')
        dpd_output_directory = params.pop('dpd_output_directory', None)
        max_dpd_logical_forms = params.pop('max_dpd_logical_forms', 10)
        max_dpd_tries = params.pop('max_dpd_tries', 20)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        table_token_indexers = TokenIndexer.dict_from_params(params.pop('table_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return WikiTablesDatasetReader(tables_directory=tables_directory,
                                       dpd_output_directory=dpd_output_directory,
                                       max_dpd_logical_forms=max_dpd_logical_forms,
                                       max_dpd_tries=max_dpd_tries,
                                       tokenizer=tokenizer,
                                       question_token_indexers=question_token_indexers,
                                       table_token_indexers=table_token_indexers)
