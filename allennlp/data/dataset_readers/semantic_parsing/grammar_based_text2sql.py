from typing import Dict, List
import logging
import json
import glob
import os
import sqlite3

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, ProductionRuleField, ListField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_utils import text2sql_utils
from allennlp.semparse.worlds.text2sql_world import Text2SqlWorld
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import read_dataset_schema

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("grammar_based_text2sql")
class GrammarBasedText2SqlDatasetReader(DatasetReader):
    """
    Reads text2sql data from
    `"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
    for a type constrained semantic parser.

    Parameters
    ----------
    schema_path : ``str``, required.
        The path to the database schema.
    database_path : ``str``, optional (default = None)
        The path to a database.
    use_all_sql : ``bool``, optional (default = False)
        Whether to use all of the sql queries which have identical semantics,
        or whether to just use the first one.
    remove_unneeded_aliases : ``bool``, (default = True)
        Whether or not to remove table aliases in the SQL which
        are not required.
    use_prelinked_entities : ``bool``, (default = True)
        Whether or not to use the pre-linked entities in the text2sql data.
    use_untyped_entities : ``bool``, (default = True)
        Whether or not to attempt to infer the pre-linked entity types.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    cross_validation_split_to_exclude : ``int``, optional (default = None)
        Some of the text2sql datasets are very small, so you may need to do cross validation.
        Here, you can specify a integer corresponding to a split_{int}.json file not to include
        in the training set.
    keep_if_unparsable : ``bool``, optional (default = True)
        Whether or not to keep examples that we can't parse using the grammar.
    """
    def __init__(self,
                 schema_path: str,
                 database_file: str = None,
                 use_all_sql: bool = False,
                 remove_unneeded_aliases: bool = True,
                 use_prelinked_entities: bool = True,
                 use_untyped_entities: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 cross_validation_split_to_exclude: int = None,
                 keep_if_unparseable: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_all_sql = use_all_sql
        self._remove_unneeded_aliases = remove_unneeded_aliases
        self._use_prelinked_entities = use_prelinked_entities
        self._keep_if_unparsable = keep_if_unparseable

        if not self._use_prelinked_entities:
            raise ConfigurationError("The grammar based text2sql dataset reader "
                                     "currently requires the use of entity pre-linking.")

        self._cross_validation_split_to_exclude = str(cross_validation_split_to_exclude)

        if database_file is not None:
            database_file = cached_path(database_file)
            connection = sqlite3.connect(database_file)
            self._cursor = connection.cursor()
        else:
            self._cursor = None

        self._schema_path = schema_path
        self._world = Text2SqlWorld(schema_path,
                                    self._cursor,
                                    use_prelinked_entities=use_prelinked_entities,
                                    use_untyped_entities=use_untyped_entities)

    @overrides
    def _read(self, file_path: str):
        """
        This dataset reader consumes the data from
        https://github.com/jkkummerfeld/text2sql-data/tree/master/data
        formatted using ``scripts/reformat_text2sql_data.py``.

        Parameters
        ----------
        file_path : ``str``, required.
            For this dataset reader, file_path can either be a path to a file `or` a
            path to a directory containing json files. The reason for this is because
            some of the text2sql datasets require cross validation, which means they are split
            up into many small files, for which you only want to exclude one.
        """
        files = [p for p in glob.glob(file_path)
                 if self._cross_validation_split_to_exclude not in os.path.basename(p)]
        schema = read_dataset_schema(self._schema_path)

        for path in files:
            with open(cached_path(path), "r") as data_file:
                data = json.load(data_file)

            for sql_data in text2sql_utils.process_sql_data(data,
                                                            use_all_sql=self._use_all_sql,
                                                            remove_unneeded_aliases=self._remove_unneeded_aliases,
                                                            schema=schema):
                linked_entities = sql_data.sql_variables if self._use_prelinked_entities else None
                instance = self.text_to_instance(sql_data.text_with_variables, linked_entities, sql_data.sql)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         query: List[str],
                         prelinked_entities: Dict[str, Dict[str, str]] = None,
                         sql: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(t) for t in query], self._token_indexers)
        fields["tokens"] = tokens

        if sql is not None:
            action_sequence, all_actions = self._world.get_action_sequence_and_all_actions(sql,
                                                                                           prelinked_entities)
            if action_sequence is None and self._keep_if_unparsable:
                print("Parse error")
                action_sequence = []
            elif action_sequence is None:
                return None

        index_fields: List[Field] = []
        production_rule_fields: List[Field] = []

        for production_rule in all_actions:
            nonterminal, _ = production_rule.split(' ->')
            production_rule = ' '.join(production_rule.split(' '))
            field = ProductionRuleField(production_rule,
                                        self._world.is_global_rule(nonterminal),
                                        nonterminal=nonterminal)
            production_rule_fields.append(field)

        valid_actions_field = ListField(production_rule_fields)
        fields["valid_actions"] = valid_actions_field

        action_map = {action.rule: i # type: ignore
                      for i, action in enumerate(valid_actions_field.field_list)}

        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
        if not action_sequence:
            index_fields = [IndexField(-1, valid_actions_field)]

        action_sequence_field = ListField(index_fields)
        fields["action_sequence"] = action_sequence_field
        return Instance(fields)
