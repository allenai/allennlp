# pylint: disable=invalid-name,no-self-use,protected-access
from flaky import flaky

from allennlp.common.testing import ModelTestCase
from allennlp.semparse.contexts.sql_table_context import action_sequence_to_sql

class AtisSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(AtisSemanticParserTest, self).setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "atis" / "experiment.json"),
                          str(self.FIXTURES_ROOT / "data" / "atis" / "sample.json"))

    @flaky
    def test_atis_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_action_sequence_to_sql(self):
        action_sequence = ['statement -> [query, ";"]',
                           'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                           'where_clause, ")"]',
                           'where_clause -> ["WHERE", "(", conditions, ")"]',
                           'conditions -> [condition]',
                           'condition -> [biexpr]',
                           'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                           'city_city_name_string -> ["\'BOSTON\'"]',
                           'binaryop -> ["="]',
                           'table_refs -> [table_name]',
                           'table_name -> ["city"]',
                           'select_results -> [col_refs]',
                           'col_refs -> [col_ref, ",", col_refs]',
                           'col_refs -> [col_ref]',
                           'col_ref -> ["city", ".", "city_name"]',
                           'col_ref -> ["city", ".", "city_code"]',
                           'distinct -> ["DISTINCT"]']

        sql_query = action_sequence_to_sql(action_sequence)
        assert sql_query == "( SELECT DISTINCT city . city_code , city . city_name " \
                            "FROM city WHERE ( city . city_name = 'BOSTON' ) ) ;"
