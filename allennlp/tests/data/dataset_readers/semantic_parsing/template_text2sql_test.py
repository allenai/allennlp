# pylint: disable=no-self-use,invalid-name,line-too-long

from allennlp.data.dataset_readers import TemplateText2SqlDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestTemplateText2SqlDatasetReader(AllenNlpTestCase):
    def test_reader(self):
        reader = TemplateText2SqlDatasetReader()

        instances = reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / "data" / "text2sql" / "*"))
        instances = ensure_list(instances)

        fields = instances[0].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        tags = fields["slot_tags"].labels

        assert tokens == ['how', 'many', 'buttercup', 'kitchen', 'are', 'there', 'in', 'san', 'francisco', '?']
        assert tags == ['O', 'O', 'name0', 'name0', 'O', 'O', 'O', 'city_name0', 'city_name0', 'O']
        assert fields["template"].label == "SELECT COUNT ( * ) FROM LOCATION AS LOCATIONalias0 , RESTAURANT " \
                                           "AS RESTAURANTalias0 WHERE LOCATIONalias0.CITY_NAME = 'city_name0' " \
                                           "AND RESTAURANTalias0.ID = LOCATIONalias0.RESTAURANT_ID AND " \
                                           "RESTAURANTalias0.NAME = 'name0' ;"

        fields = instances[1].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        tags = fields["slot_tags"].labels
        assert tokens == ['how', 'many', 'chinese', 'restaurants', 'are', 'there', 'in', 'the', 'bay', 'area', '?']
        assert tags == ['O', 'O', 'food_type0', 'O', 'O', 'O', 'O', 'O', 'region0', 'region0', 'O']
        assert fields["template"].label == "SELECT COUNT ( * ) FROM GEOGRAPHIC AS GEOGRAPHICalias0 , RESTAURANT AS "\
                                           "RESTAURANTalias0 WHERE GEOGRAPHICalias0.REGION = 'region0' AND "\
                                           "RESTAURANTalias0.CITY_NAME = GEOGRAPHICalias0.CITY_NAME AND "\
                                           "RESTAURANTalias0.FOOD_TYPE = 'food_type0' ;"

        fields = instances[2].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        tags = fields["slot_tags"].labels
        assert tokens == ['how', 'many', 'places', 'for', 'chinese', 'food', 'are', 'there', 'in', 'the', 'bay', 'area', '?']
        assert tags == ['O', 'O', 'O', 'O', 'food_type0', 'O', 'O', 'O', 'O', 'O', 'region0', 'region0', 'O']
        assert fields["template"].label == "SELECT COUNT ( * ) FROM GEOGRAPHIC AS GEOGRAPHICalias0 , RESTAURANT AS "\
                                           "RESTAURANTalias0 WHERE GEOGRAPHICalias0.REGION = 'region0' AND "\
                                           "RESTAURANTalias0.CITY_NAME = GEOGRAPHICalias0.CITY_NAME AND "\
                                           "RESTAURANTalias0.FOOD_TYPE = 'food_type0' ;"

        fields = instances[3].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        tags = fields["slot_tags"].labels
        assert tokens == ['how', 'many', 'chinese', 'places', 'are', 'there', 'in', 'the', 'bay', 'area', '?']
        assert tags == ['O', 'O', 'food_type0', 'O', 'O', 'O', 'O', 'O', 'region0', 'region0', 'O']
        assert fields["template"].label == "SELECT COUNT ( * ) FROM GEOGRAPHIC AS GEOGRAPHICalias0 , RESTAURANT AS "\
                                           "RESTAURANTalias0 WHERE GEOGRAPHICalias0.REGION = 'region0' AND "\
                                           "RESTAURANTalias0.CITY_NAME = GEOGRAPHICalias0.CITY_NAME AND "\
                                           "RESTAURANTalias0.FOOD_TYPE = 'food_type0' ;"
        fields = instances[4].fields
        tokens = [t.text for t in fields["tokens"].tokens]
        tags = fields["slot_tags"].labels
        assert tokens == ['how', 'many', 'places', 'for', 'chinese', 'are', 'there', 'in', 'the', 'bay', 'area', '?']
        assert tags == ['O', 'O', 'O', 'O', 'food_type0', 'O', 'O', 'O', 'O', 'region0', 'region0', 'O']
        assert fields["template"].label == "SELECT COUNT ( * ) FROM GEOGRAPHIC AS GEOGRAPHICalias0 , RESTAURANT AS "\
                                           "RESTAURANTalias0 WHERE GEOGRAPHICalias0.REGION = 'region0' AND "\
                                           "RESTAURANTalias0.CITY_NAME = GEOGRAPHICalias0.CITY_NAME AND "\
                                           "RESTAURANTalias0.FOOD_TYPE = 'food_type0' ;"
