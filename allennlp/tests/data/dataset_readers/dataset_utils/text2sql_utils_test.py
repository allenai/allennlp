# pylint: disable=no-self-use,invalid-name,line-too-long

import json
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils import text2sql_utils

class Text2SqlUtilsTest(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        self.data = AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'text2sql' / 'restaurants_tiny.json'

    def test_process_sql_data_blob(self):

        data = json.load(open(str(self.data)))
        dataset = text2sql_utils.process_sql_data_blob(data[0])

        # All of these data points are the same. This is weird, but currently correct.
        for sql_data in dataset:
            assert sql_data.text == ['how', 'many', 'buttercup', 'kitchen', 'are', 'there', 'in', 'san', 'francisco', '?']
            assert sql_data.text_with_variables == ['how', 'many', 'name0', 'are', 'there', 'in', 'city_name0', '?']
            assert sql_data.sql == ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', 'AS', 'LOCATIONalias0', ',',
                                    'RESTAURANT', 'AS', 'RESTAURANTalias0', 'WHERE', 'LOCATIONalias0.CITY_NAME', '=',
                                    'city_name0', 'AND', 'RESTAURANTalias0.ID', '=', 'LOCATIONalias0.RESTAURANT_ID',
                                    'AND', 'RESTAURANTalias0.NAME', '=', 'name0', ';']
            assert sql_data.text_variables == {'city_name0': 'san francisco', 'name0': 'buttercup kitchen'}
            assert sql_data.sql_variables == {'city_name0': 'san francisco', 'name0': 'buttercup kitchen'}


        # This section of the dataset should be in the train set because it's not used for cross validation.
        dataset = text2sql_utils.process_sql_data_blob(data[1])

        correct_text = [
                [['how', 'many', 'chinese', 'restaurants', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'restaurants', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'chinese', 'restaurants', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'restaurants', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'places', 'for', 'chinese', 'food', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'places', 'for', 'food_type0', 'food', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'chinese', 'places', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'places', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'places', 'for', 'chinese', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'places', 'for', 'food_type0', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'chinese', 'restaurants', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'restaurants', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'chinese', 'restaurants', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'restaurants', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'places', 'for', 'chinese', 'food', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'places', 'for', 'food_type0', 'food', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'chinese', 'places', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'places', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'places', 'for', 'chinese', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'places', 'for', 'food_type0', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'chinese', 'restaurants', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'restaurants', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'chinese', 'restaurants', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'restaurants', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'places', 'for', 'chinese', 'food', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'places', 'for', 'food_type0', 'food', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'chinese', 'places', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'food_type0', 'places', 'are', 'there', 'in', 'the', 'region0', '?']],
                [['how', 'many', 'places', 'for', 'chinese', 'are', 'there', 'in', 'the', 'bay', 'area', '?'],
                 ['how', 'many', 'places', 'for', 'food_type0', 'are', 'there', 'in', 'the', 'region0', '?']]
        ]

        for i, sql_data in enumerate(dataset):
            assert sql_data.sql == ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'GEOGRAPHIC', 'AS', 'GEOGRAPHICalias0',
                                    ',', 'RESTAURANT', 'AS', 'RESTAURANTalias0', 'WHERE', 'GEOGRAPHICalias0.REGION',
                                    '=', 'region0', 'AND', 'RESTAURANTalias0.CITY_NAME', '=', 'GEOGRAPHICalias0.CITY_NAME',
                                    'AND', 'RESTAURANTalias0.FOOD_TYPE', '=', 'food_type0', ';']
            assert sql_data.text_variables == {'region0': 'bay area', 'food_type0': 'chinese'}
            assert sql_data.sql_variables == {'region0': 'bay area', 'food_type0': 'chinese'}
            assert sql_data.text == correct_text[i][0]
            assert sql_data.text_with_variables == correct_text[i][1]

    def test_replace_variables(self):
        sentence = ['how', 'many', 'name0', 'are', 'there', 'in', 'city_name0', '?']
        sentence_variables = {'city_name0': 'san francisco', 'name0': 'buttercup kitchen'}
        assert text2sql_utils.replace_variables(sentence, sentence_variables) == ['how', 'many', 'buttercup', 'kitchen',
                                                                                  'are', 'there', 'in', 'san', 'francisco', '?']

    def test_clean_and_split_sql(self):
        sql = "SELECT COUNT( * ) FROM LOCATION AS LOCATIONalias0 , RESTAURANT AS RESTAURANTalias0 WHERE LOCATIONalias0.CITY_NAME " \
              "= \"city_name0\" AND RESTAURANTalias0.ID = LOCATIONalias0.RESTAURANT_ID AND RESTAURANTalias0.NAME = \"name0\" ;"

        cleaned = text2sql_utils.clean_and_split_sql(sql)
        assert cleaned == ['SELECT', 'COUNT', '(', '*', ')', 'FROM', 'LOCATION', 'AS', 'LOCATIONalias0', ',',
                           'RESTAURANT', 'AS', 'RESTAURANTalias0', 'WHERE', 'LOCATIONalias0.CITY_NAME', '=',
                           'city_name0', 'AND', 'RESTAURANTalias0.ID', '=', 'LOCATIONalias0.RESTAURANT_ID',
                           'AND', 'RESTAURANTalias0.NAME', '=', 'name0', ';']
