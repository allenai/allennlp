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

        assert indices == [93, 75, 78, 88, 86, 82, 39, 113, 48, 42, 2, 46, 91, 90, 102, 92, 90, 103, 118, 34, 5,
                           112, 21, 102, 23, 13, 34, 5, 116, 95, 16, 34, 5, 112, 21, 103, 30, 13, 34, 5, 112, 21,
                           102, 30, 16, 34, 5, 112, 21, 103, 27, 13, 39, 116, 96]

        action_fields = fields["valid_actions"].field_list
        production_rules = [(x.rule, x.is_global_rule) for x in action_fields]

        assert production_rules == [('arg_list -> [expr, ",", arg_list]', True) ,
                                    ('arg_list -> [expr]', True) ,
                                    ('arg_list_or_star -> ["*"]', True) ,
                                    ('arg_list_or_star -> [arg_list]', True) ,
                                    ('between_expr -> [value, "BETWEEN", value, "AND", value]', True) ,
                                    ('binary_expr -> [value, binaryop, expr]', True) ,
                                    ('binaryop -> ["*"]', True) ,
                                    ('binaryop -> ["+"]', True) ,
                                    ('binaryop -> ["-"]', True) ,
                                    ('binaryop -> ["/"]', True) ,
                                    ('binaryop -> ["<"]', True) ,
                                    ('binaryop -> ["<="]', True) ,
                                    ('binaryop -> ["<>"]', True) ,
                                    ('binaryop -> ["="]', True) ,
                                    ('binaryop -> [">"]', True) ,
                                    ('binaryop -> [">="]', True) ,
                                    ('binaryop -> ["AND"]', True) ,
                                    ('binaryop -> ["LIKE"]', True) ,
                                    ('binaryop -> ["OR"]', True) ,
                                    ('boolean -> ["false"]', True) ,
                                    ('boolean -> ["true"]', True) ,
                                    ('col_ref -> [table_name, ".", column_name]', True) ,
                                    ('col_ref -> [table_name]', True) ,
                                    ('column_name -> ["CITY_NAME"]', True) ,
                                    ('column_name -> ["COUNTY"]', True) ,
                                    ('column_name -> ["FOOD_TYPE"]', True) ,
                                    ('column_name -> ["HOUSE_NUMBER"]', True) ,
                                    ('column_name -> ["NAME"]', True) ,
                                    ('column_name -> ["RATING"]', True) ,
                                    ('column_name -> ["REGION"]', True) ,
                                    ('column_name -> ["RESTAURANT_ID"]', True) ,
                                    ('column_name -> ["STREET_NAME"]', True) ,
                                    ('expr -> ["(", query, ")"]', True) ,
                                    ('expr -> [between_expr]', True) ,
                                    ('expr -> [binary_expr]', True) ,
                                    ('expr -> [in_expr]', True) ,
                                    ('expr -> [like_expr]', True) ,
                                    ('expr -> [null_check_expr]', True) ,
                                    ('expr -> [unary_expr]', True) ,
                                    ('expr -> [value]', True) ,
                                    ('fname -> ["ALL"]', True) ,
                                    ('fname -> ["AVG"]', True) ,
                                    ('fname -> ["COUNT"]', True) ,
                                    ('fname -> ["MAX"]', True) ,
                                    ('fname -> ["MIN"]', True) ,
                                    ('fname -> ["SUM"]', True) ,
                                    ('from_clause -> ["FROM", source]', True) ,
                                    ('function -> [fname, "(", "DISTINCT", arg_list_or_star, ")"]', True) ,
                                    ('function -> [fname, "(", arg_list_or_star, ")"]', True) ,
                                    ('group_clause -> [expr, "," group_clause]', True) ,
                                    ('group_clause -> [expr]', True) ,
                                    ('groupby_clause -> ["GROUP", "BY" group_clause, "HAVING", expr]', True) ,
                                    ('groupby_clause -> ["GROUP", "BY" group_clause]', True) ,
                                    ('in_expr -> [value, "IN", expr]', True) ,
                                    ('in_expr -> [value, "IN", string_set]', True) ,
                                    ('in_expr -> [value, "NOT", "IN", expr]', True) ,
                                    ('in_expr -> [value, "NOT", "IN", string_set]', True) ,
                                    ('like_expr -> [value, "LIKE", string]', True) ,
                                    ('limit -> ["LIMIT", number]', True) ,
                                    ('null_check_expr -> [col_ref, "IS", "NOT", "NULL"]', True) ,
                                    ('null_check_expr -> [col_ref, "IS", "NULL"]', True) ,
                                    ('order_clause -> [ordering_term, "," order_clause]', True) ,
                                    ('order_clause -> [ordering_term]', True) ,
                                    ('orderby_clause -> ["ORDER", "BY" order_clause]', True) ,
                                    ('ordering -> ["ASC"]', True) ,
                                    ('ordering -> ["DESC"]', True) ,
                                    ('ordering_term -> [expr ordering]', True) ,
                                    ('ordering_term -> [expr]', True) ,
                                    ('parenval -> ["(", expr, ")"]', True) ,
                                    ('query -> [select_core groupby_clause, limit]', True) ,
                                    ('query -> [select_core groupby_clause, orderby_clause, limit]', True) ,
                                    ('query -> [select_core groupby_clause, orderby_clause]', True) ,
                                    ('query -> [select_core groupby_clause]', True) ,
                                    ('query -> [select_core orderby_clause, limit]', True) ,
                                    ('query -> [select_core orderby_clause]', True) ,
                                    ('query -> [select_core]', True) ,
                                    ("sel_res_all_star -> ['*']", True) ,
                                    ('sel_res_tab_star -> [table_name ".*"]', True) ,
                                    ('select_core -> [select_with_distinct select_results from_clause where_clause]', True) ,
                                    ('select_core -> [select_with_distinct select_results from_clause]', True) ,
                                    ('select_core -> [select_with_distinct select_results where_clause]', True) ,
                                    ('select_core -> [select_with_distinct select_results]', True) ,
                                    ('select_result -> [expr]', True) ,
                                    ('select_result -> [sel_res_all_star]', True) ,
                                    ('select_result -> [sel_res_tab_star]', True) ,
                                    ('select_results -> [select_result, ",", select_results]', True) ,
                                    ('select_results -> [select_result]', True) ,
                                    ('select_with_distinct -> ["SELECT", "DISTINCT"]', True) ,
                                    ('select_with_distinct -> ["SELECT"]', True) ,
                                    ('single_source -> ["(", query, ")"]', True) ,
                                    ('single_source -> [table_name]', True) ,
                                    ('source -> [single_source, ",", source]', True) ,
                                    ('source -> [single_source]', True) ,
                                    ('statement -> [query, ";"]', True) ,
                                    ('statement -> [query]', True) ,
                                    ('string -> ["\'city_name0\'"]', True) ,
                                    ('string -> ["\'name0\'"]', True) ,
                                    ('string -> [~"\'.*?\'"iu]', True) ,
                                    ('string_set -> ["(", string_set_vals, ")"]', True) ,
                                    ('string_set_vals -> [string, ",", string_set_vals]', True) ,
                                    ('string_set_vals -> [string]', True) ,
                                    ('table_name -> ["GEOGRAPHIC"]', True) ,
                                    ('table_name -> ["LOCATION"]', True) ,
                                    ('table_name -> ["RESTAURANT"]', True) ,
                                    ('unary_expr -> [unaryop expr]', True) ,
                                    ('unaryop -> ["+"]', True) ,
                                    ('unaryop -> ["-"]', True) ,
                                    ('unaryop -> ["NOT"]', True) ,
                                    ('unaryop -> ["not"]', True) ,
                                    ('value -> ["2.5"]', True) ,
                                    ('value -> ["YEAR(CURDATE())"]', True) ,
                                    ('value -> [boolean]', True) ,
                                    ('value -> [col_ref]', True) ,
                                    ('value -> [function]', True) ,
                                    ('value -> [number]', True) ,
                                    ('value -> [parenval]', True) ,
                                    ('value -> [string]', True) ,
                                    ('where_clause -> ["WHERE", expr where_conj]', True) ,
                                    ('where_clause -> ["WHERE", expr]', True) ,
                                    ('where_conj -> ["AND", expr where_conj]', True) ,
                                    ('where_conj -> ["AND", expr]', True)]
