from typing import Dict, List
import logging
import json
import glob
import os

from overrides import overrides
from parsimonious.exceptions import ParseError

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, ProductionRuleField, ListField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_utils import text2sql_utils
from allennlp.semparse.contexts.text2sql_table_context import Text2SqlTableContext
from allennlp.semparse.worlds.text2sql_world import Text2SqlWorld

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
    use_all_sql : ``bool``, optional (default = False)
        Whether to use all of the sql queries which have identical semantics,
        or whether to just use the first one.
    remove_unneeded_aliases : ``bool``, (default = True)
        Whether or not to remove table aliases in the SQL which
        are not required.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    cross_validation_split_to_exclude : ``int``, optional (default = None)
        Some of the text2sql datasets are very small, so you may need to do cross validation.
        Here, you can specify a integer corresponding to a split_{int}.json file not to include
        in the training set.
    """
    def __init__(self,
                 schema_path: str,
                 use_all_sql: bool = False,
                 remove_unneeded_aliases: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 cross_validation_split_to_exclude: int = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_all_sql = use_all_sql
        self._remove_unneeded_aliases = remove_unneeded_aliases
        self._cross_validation_split_to_exclude = str(cross_validation_split_to_exclude)

        self._sql_table_context = Text2SqlTableContext(schema_path)
        self._world = Text2SqlWorld(self._sql_table_context)

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

        for path in files:
            with open(cached_path(path), "r") as data_file:
                data = json.load(data_file)

            for sql_data in text2sql_utils.process_sql_data(data,
                                                            use_all_sql=self._use_all_sql,
                                                            remove_unneeded_aliases=self._remove_unneeded_aliases,
                                                            # TODO(Mark): Horrible hack, remove
                                                            schema=self._sql_table_context.schema):
                instance = self.text_to_instance(sql_data.text, sql_data)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         query: List[str],
                         sql: text2sql_utils.SqlData = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(t) for t in query], self._token_indexers)
        fields["tokens"] = tokens

        if sql is not None:
            try:
                action_sequence, all_actions = self._world.get_action_sequence_and_all_actions(sql.sql)
            except ParseError:
                return None

        index_fields: List[Field] = []
        production_rule_fields: List[Field] = []

        for production_rule in all_actions:
            nonterminal, _ = production_rule.split(' ->')
            production_rule = ' '.join(production_rule.split(' '))
            field = ProductionRuleField(production_rule, self._world.is_global_rule(nonterminal))
            production_rule_fields.append(field)

        valid_actions_field = ListField(production_rule_fields)
        fields["valid_actions"] = valid_actions_field

        action_map = {action.rule: i # type: ignore
                      for i, action in enumerate(valid_actions_field.field_list)}

        for production_rule in action_sequence:
            # Temporarily skipping this production to
            # make a PR smaller. The next PR will constrain
            # the strings produced to be from the table,
            # but at the moment they are blank so they
            # aren't present in the global actions.
            # TODO(Mark): fix the above.
            if production_rule.startswith("string"):
                continue
            index_fields.append(IndexField(action_map[production_rule], valid_actions_field))

        action_sequence_field = ListField(index_fields)
        fields["action_sequence"] = action_sequence_field
        return Instance(fields)
