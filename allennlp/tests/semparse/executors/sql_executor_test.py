from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.executors import SqlExecutor

class SqlExecutorTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self._database_file = "https://allennlp.s3.amazonaws.com/datasets/atis/atis.db"

    def test_sql_accuracy_is_scored_correctly(self):
        sql_query_label = ("( SELECT airport_service . airport_code "
                           "FROM airport_service "
                           "WHERE airport_service . city_code IN ( "
                           "SELECT city . city_code FROM city "
                           "WHERE city.city_name = 'BOSTON' ) ) ;")

        executor = SqlExecutor(self._database_file)
        postprocessed_sql_query_label = executor.postprocess_query_sqlite(sql_query_label)
        # If the predicted query and the label are the same, then we should get 1.
        assert executor.evaluate_sql_query(postprocessed_sql_query_label,
                                           [postprocessed_sql_query_label]) == 1

        predicted_sql_query = ("( SELECT airport_service . airport_code "
                               "FROM airport_service "
                               "WHERE airport_service . city_code IN ( "
                               "SELECT city . city_code FROM city "
                               "WHERE city.city_name = 'SEATTLE' ) ) ;")

        postprocessed_predicted_sql_query = executor.postprocess_query_sqlite(predicted_sql_query)
        # If the predicted query and the label are different we should get 0.
        assert executor.evaluate_sql_query(postprocessed_predicted_sql_query,
                                           [postprocessed_sql_query_label]) == 0
