# pylint: disable=invalid-name,line-too-long
from allennlp.data.dataset_readers.semantic_parsing.grammar_based_text2sql import GrammarBasedText2SqlDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestGrammarBasdText2SqlDatasetReader(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.data_path = str(self.FIXTURES_ROOT / 'data' / 'text2sql'/ '*.json')
        self.schema = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants-schema.csv')
        self.database = str(self.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants.db')

        self.reader = GrammarBasedText2SqlDatasetReader(self.schema, self.database)

    def test_reader_can_read_data_with_entity_pre_linking(self):
        instances = self.reader.read(self.data_path)
        instances = list(instances)
        assert len(instances) == 5

        fields = instances[0].fields
        token_field = fields["tokens"]
        tokens = [t.text for t in token_field.tokens]
        assert tokens == ['how', 'many', 'buttercup', 'kitchen', 'are', 'there', 'in', 'san', 'francisco', '?']

        action_sequence = fields["action_sequence"].field_list
        indices = [x.sequence_index for x in action_sequence]
        assert indices == [101, 75, 81, 124, 33, 5, 33, 5, 33, 5, 33, 5, 33, 5,
                           39, 115, 13, 118, 21, 27, 108, 16, 118, 21, 30, 107, 13,
                           118, 21, 30, 108, 16, 114, 13, 118, 21, 23, 107, 46, 95,
                           96, 94, 100, 108, 94, 100, 107, 90, 88, 80, 39, 119,
                           48, 2, 42, 92]

        action_fields = fields["valid_actions"].field_list
        production_rules = [(x.rule, x.is_global_rule) for x in action_fields]

        assert production_rules == [('arg_list -> [expr, ",", arg_list]', True),
                                    ('arg_list -> [expr]', True),
                                    ('arg_list_or_star -> ["*"]', True),
                                    ('arg_list_or_star -> [arg_list]', True),
                                    ('between_expr -> [value, "BETWEEN", value, "AND", value]', True),
                                    ('binary_expr -> [value, binaryop, expr]', True),
                                    ('binaryop -> ["*"]', True),
                                    ('binaryop -> ["+"]', True),
                                    ('binaryop -> ["-"]', True),
                                    ('binaryop -> ["/"]', True),
                                    ('binaryop -> ["<"]', True),
                                    ('binaryop -> ["<="]', True),
                                    ('binaryop -> ["<>"]', True),
                                    ('binaryop -> ["="]', True),
                                    ('binaryop -> [">"]', True),
                                    ('binaryop -> [">="]', True),
                                    ('binaryop -> ["AND"]', True),
                                    ('binaryop -> ["LIKE"]', True),
                                    ('binaryop -> ["OR"]', True),
                                    ('boolean -> ["false"]', True),
                                    ('boolean -> ["true"]', True),
                                    ('col_ref -> [table_name, ".", column_name]', True),
                                    ('col_ref -> [table_name]', True),
                                    ('column_name -> ["CITY_NAME"]', True),
                                    ('column_name -> ["COUNTY"]', True),
                                    ('column_name -> ["FOOD_TYPE"]', True),
                                    ('column_name -> ["HOUSE_NUMBER"]', True),
                                    ('column_name -> ["NAME"]', True),
                                    ('column_name -> ["RATING"]', True),
                                    ('column_name -> ["REGION"]', True),
                                    ('column_name -> ["RESTAURANT_ID"]', True),
                                    ('column_name -> ["STREET_NAME"]', True),
                                    ('expr -> [between_expr]', True),
                                    ('expr -> [binary_expr]', True),
                                    ('expr -> [in_expr]', True),
                                    ('expr -> [like_expr]', True),
                                    ('expr -> [null_check_expr]', True),
                                    ('expr -> [source_subq]', True),
                                    ('expr -> [unary_expr]', True),
                                    ('expr -> [value]', True),
                                    ('fname -> ["ALL"]', True),
                                    ('fname -> ["AVG"]', True),
                                    ('fname -> ["COUNT"]', True),
                                    ('fname -> ["MAX"]', True),
                                    ('fname -> ["MIN"]', True),
                                    ('fname -> ["SUM"]', True),
                                    ('from_clause -> ["FROM", source]', True),
                                    ('function -> [fname, "(", "DISTINCT", arg_list_or_star, ")"]', True),
                                    ('function -> [fname, "(", arg_list_or_star, ")"]', True),
                                    ('group_clause -> [expr, "," group_clause]', True),
                                    ('group_clause -> [expr]', True),
                                    ('groupby_clause -> ["GROUP", "BY" group_clause, "HAVING", expr]', True),
                                    ('groupby_clause -> ["GROUP", "BY" group_clause]', True),
                                    ('in_expr -> [value, "IN", expr]', True),
                                    ('in_expr -> [value, "IN", string_set]', True),
                                    ('in_expr -> [value, "NOT", "IN", expr]', True),
                                    ('in_expr -> [value, "NOT", "IN", string_set]', True),
                                    ('like_expr -> [value, "LIKE", string]', True),
                                    ('limit -> ["LIMIT", number]', True),
                                    ('null_check_expr -> [col_ref, "IS", "NOT", "NULL"]', True),
                                    ('null_check_expr -> [col_ref, "IS", "NULL"]', True),
                                    ('order_clause -> [ordering_term, "," order_clause]', True),
                                    ('order_clause -> [ordering_term]', True),
                                    ('orderby_clause -> ["ORDER", "BY" order_clause]', True),
                                    ('ordering -> ["ASC"]', True), ('ordering -> ["DESC"]', True),
                                    ('ordering_term -> [expr ordering]', True),
                                    ('ordering_term -> [expr]', True),
                                    ('parenval -> ["(", expr, ")"]', True),
                                    ('query -> [select_core groupby_clause, limit]', True),
                                    ('query -> [select_core groupby_clause, orderby_clause, limit]', True),
                                    ('query -> [select_core groupby_clause, orderby_clause]', True),
                                    ('query -> [select_core groupby_clause]', True),
                                    ('query -> [select_core orderby_clause, limit]', True),
                                    ('query -> [select_core orderby_clause]', True),
                                    ('query -> [select_core]', True),
                                    ("sel_res_all_star -> ['*']", True),
                                    ('sel_res_col -> [col_ref, "AS", name]', True),
                                    ('sel_res_tab_star -> [name ".*"]', True),
                                    ('sel_res_val -> [expr, "AS", name]', True),
                                    ('sel_res_val -> [expr]', True),
                                    ('select_core -> [select_with_distinct select_results from_clause where_clause]', True),
                                    ('select_core -> [select_with_distinct select_results from_clause]', True),
                                    ('select_core -> [select_with_distinct select_results where_clause]', True),
                                    ('select_core -> [select_with_distinct select_results]', True),
                                    ('select_result -> [sel_res_all_star]', True), ('select_result -> [sel_res_col]', True),
                                    ('select_result -> [sel_res_tab_star]', True), ('select_result -> [sel_res_val]', True),
                                    ('select_results -> [select_result, ",", select_results]', True),
                                    ('select_results -> [select_result]', True),
                                    ('select_with_distinct -> ["SELECT", "DISTINCT"]', True),
                                    ('select_with_distinct -> ["SELECT"]', True),
                                    ('single_source -> [source_subq]', True),
                                    ('single_source -> [source_table]', True),
                                    ('source -> [single_source, ",", source]', True),
                                    ('source -> [single_source]', True),
                                    ('source_subq -> ["(", query, ")", "AS", name]', True),
                                    ('source_subq -> ["(", query, ")"]', True),
                                    ('source_table -> [table_name, "AS", name]', True),
                                    ('source_table -> [table_name]', True),
                                    ('statement -> [query, ";"]', True),
                                    ('statement -> [query]', True),
                                    ('string_set -> ["(", string_set_vals, ")"]', True),
                                    ('string_set_vals -> [string, ",", string_set_vals]', True),
                                    ('string_set_vals -> [string]', True),
                                    ('table_name -> ["GEOGRAPHIC"]', True),
                                    ('table_name -> ["LOCATION"]', True),
                                    ('table_name -> ["RESTAURANT"]', True),
                                    ('unary_expr -> [unaryop expr]', True),
                                    ('unaryop -> ["+"]', True),
                                    ('unaryop -> ["-"]', True),
                                    ('unaryop -> ["NOT"]', True),
                                    ('unaryop -> ["not"]', True),
                                    ('value -> ["\'city_name0\'"]', True),
                                    ('value -> ["\'name0\'"]', True),
                                    ('value -> ["YEAR(CURDATE())"]', True),
                                    ('value -> [boolean]', True),
                                    ('value -> [col_ref]', True),
                                    ('value -> [function]', True),
                                    ('value -> [number]', True),
                                    ('value -> [parenval]', True),
                                    ('value -> [string]', True),
                                    ('where_clause -> ["WHERE", expr where_conj]', True),
                                    ('where_clause -> ["WHERE", expr]', True),
                                    ('where_conj -> ["AND", expr where_conj]', True),
                                    ('where_conj -> ["AND", expr]', True)]
