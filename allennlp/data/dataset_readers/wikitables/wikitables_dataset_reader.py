"""
Reader for WikitableQuestions (https://github.com/ppasupat/WikiTableQuestions/releases/tag/v1.0.2).
"""

from typing import Any, Dict, List, Union
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
from allennlp.data.semparse.type_declarations import type_declaration as types
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
                 tokenizer: Tokenizer = None,
                 question_token_indexers: Dict[str, TokenIndexer] = None,
                 table_token_indexers: Dict[str, TokenIndexer] = None,
                 nonterminal_indexers: Dict[str, TokenIndexer] = None,
                 terminal_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tables_directory = tables_directory
        self._dpd_output_directory = dpd_output_directory
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
            for line in tqdm.tqdm(data_file):
                line = line.strip("\n")
                if not line:
                    continue
                parsed_info = self._parse_line_as_lisp(line)
                question = parsed_info["question"]
                # We want the TSV file, but the ``*.examples`` files typically point to CSV.
                table_filename = os.path.join(self._tables_directory,
                                              parsed_info["context"].replace(".csv", ".tsv"))
                dpd_output_filename = os.path.join(self._dpd_output_directory,
                                                   parsed_info["id"] + '.gz')
                sempre_forms = [line.strip().decode('utf-8') for line in gzip.open(dpd_output_filename)]
                instances.append(self.text_to_instance(question, table_filename, sempre_forms))
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
        table_info : ``str`` or ``Dict[str, Any]``
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
            expressions = [world.parse_logical_form(form) for form in dpd_output]
            action_sequences = [world.get_action_sequence(expression) for expression in expressions]

            # We'll make each target action sequence a List[IndexField], where the index is into
            # the action list we made above.  We need to ignore the type here because mypy doesn't
            # like `action.rule` - it's hard to tell mypy that the ListField is made up of
            # ProductionRuleFields.
            action_map = {action.rule: i for i, action in enumerate(action_field.field_list)}  # type: ignore
            action_sequence_fields: List[Field] = []
            for form, expression, sequence in zip(dpd_output, expressions, action_sequences):
                index_fields: List[Field] = []
                print(form)
                print(expression)
                print(sequence)
                for production_rule in sequence:
                    field = IndexField(action_map[production_rule], action_field)
                    index_fields.append(field)
                action_sequence_fields.append(ListField(index_fields))
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

    @classmethod
    def from_params(cls, params: Params) -> 'WikiTablesDatasetReader':
        tables_directory = params.pop('tables_directory')
        dpd_output_directory = params.pop('dpd_output_directory', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        question_token_indexers = TokenIndexer.dict_from_params(params.pop('question_token_indexers', {}))
        table_token_indexers = TokenIndexer.dict_from_params(params.pop('table_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return WikiTablesDatasetReader(tables_directory=tables_directory,
                                       dpd_output_directory=dpd_output_directory,
                                       tokenizer=tokenizer,
                                       question_token_indexers=question_token_indexers,
                                       table_token_indexers=table_token_indexers)
