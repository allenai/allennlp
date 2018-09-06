# pylint: disable=no-self-use,invalid-name,line-too-long

import json
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils import text2sql_utils

class Text2SqlUtilsTest(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        self.data = AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants_tiny.json'

    def test_process_sql_data_blob(self):
        data = json.load(open(str(self.data)))[0]
        dataset = text2sql_utils.process_sql_data_blob(data, cross_validation_split=1)

        for x in dataset:
            print(x)

    def test_get_tokens(self):
        sentence = ['how', 'many', 'name0', 'are', 'there', 'in', 'city_name0', '?']
        sentence_variables = {'city_name0': 'san francisco', 'name0': 'buttercup kitchen'}
        assert text2sql_utils.get_tokens(sentence, sentence_variables) == ['how', 'many', 'buttercup', 'kitchen',
                                                                           'are', 'there', 'in', 'san', 'francisco', '?']

    def test_clean_and_split_sql(self):
        sql = "SELECT COUNT( * ) FROM LOCATION AS LOCATIONalias0 , RESTAURANT AS RESTAURANTalias0 WHERE LOCATIONalias0.CITY_NAME " \
              "= \"city_name0\" AND RESTAURANTalias0.ID = LOCATIONalias0.RESTAURANT_ID AND RESTAURANTalias0.NAME = \"name0\" ;"

        cleaned = text2sql_utils.clean_and_split_sql(sql)
        assert cleaned == ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', 'AS', 'LOCATIONalias0', ',',
                           'RESTAURANT', 'AS', 'RESTAURANTalias0', 'WHERE', 'LOCATIONalias0.CITY_NAME', '=',
                           'city_name0', 'AND', 'RESTAURANTalias0.ID', '=', 'LOCATIONalias0.RESTAURANT_ID',
                           'AND', 'RESTAURANTalias0.NAME', '=', 'name0', ';']
